"""Playwright browser controller — lifecycle, multi-tab tracking, navigation.

Tracks every open tab in ``{tab_id: Page}``. New tabs opened by clicks
(popups / ``target=_blank``) are detected via the context ``page`` event and
become the active tab automatically; the agent can switch back or close them.

Includes CDP screencast support: ``start_screencast`` opens a DevTools Protocol
session and streams JPEG frames from the active page to a callback, giving the
desktop UI a live embedded browser view without a visible Chromium window.
"""

import asyncio
import os
import subprocess
import sys
from functools import partial
from typing import Any, Callable

from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import Stealth

import paths
from config import config

FrameCallback = Callable[[str], None]





def _chrome_user_data_dir() -> str:
    """Return the persistent Chrome profile directory used by the agent.

    Stored under the per-user writable data dir (see ``paths.chrome_profile_dir``)
    so logins / cookies survive across runs without touching the user's real
    Chrome ``User Data`` folder.
    """
    return paths.chrome_profile_dir()


def _close_chrome() -> None:
    """Terminate any running Chrome processes so the profile lock is released.

    Chrome holds an exclusive lock on its profile directory.  Playwright's
    ``launch_persistent_context`` cannot open that directory while Chrome is
    already running.  We close Chrome silently and wait briefly for the OS to
    release all file handles before Playwright tries to launch.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                capture_output=True,
                timeout=8,
            )
        elif sys.platform == "darwin":
            subprocess.run(
                ["pkill", "-x", "Google Chrome"],
                capture_output=True,
                timeout=8,
            )
        else:
            subprocess.run(
                ["pkill", "-f", "google-chrome"],
                capture_output=True,
                timeout=8,
            )
        
        import time
        time.sleep(1.5)
    except Exception:  
        pass


class BrowserController:
    """Manages a Chromium browser and all of its tabs via Playwright."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[int, Page] = {}
        self._active_id: int = 0
        self._next_id: int = 0

    

    async def start(self) -> None:
        """Launch a browser window with stealth patches.

        Uses the user's real Chrome profile when ``config.use_real_chrome`` is
        True; otherwise launches a fresh Chromium instance.
        """
        self._playwright = await async_playwright().start()

        _stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins',   {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        """

        if config.use_real_chrome:
            _close_chrome()

            profile_path = _chrome_user_data_dir()
            try:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    profile_path,
                    channel="chrome",
                    headless=config.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-session-crashed-bubble",
                        "--disable-infobars",
                        "--hide-crash-restore-bubble",
                    ],
                    viewport={
                        "width": config.viewport_width,
                        "height": config.viewport_height,
                    },
                    ignore_default_args=["--enable-automation"],
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not launch Chrome with your real profile.\n"
                    f"  Profile path : {profile_path}\n"
                    "  Make sure Google Chrome is installed and close it fully\n"
                    "  before running the agent.\n"
                    f"  Detail: {exc}"
                ) from exc
            self._browser = None
            await self._context.add_init_script(_stealth_script)
            self._context.on("page", self._register_page)
            if self._context.pages:
                first_page = self._context.pages[0]
                self._register_page(first_page)
            else:
                first_page = await self._context.new_page()
            await Stealth().apply_stealth_async(first_page)
            
            
            await first_page.goto("about:blank", wait_until="domcontentloaded", timeout=10_000)
            await asyncio.sleep(2.0)
        else:
            self._browser = await self._playwright.chromium.launch(
                headless=config.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--hide-crash-restore-bubble",
                ],
                ignore_default_args=["--enable-automation"],
            )
            self._context = await self._browser.new_context(
                viewport={
                    "width": config.viewport_width,
                    "height": config.viewport_height,
                },
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            )
            await self._context.add_init_script(_stealth_script)
            self._context.on("page", self._register_page)
            if self._context.pages:
                first_page = self._context.pages[0]
                self._register_page(first_page)
            else:
                first_page = await self._context.new_page()
            await Stealth().apply_stealth_async(first_page)
        print(f"Browser ready: {self.current_url()}")

    async def stop(self) -> None:
        """Close the browser and release all Playwright resources."""
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._pages.clear()
        self._context = None
        self._browser = None
        self._playwright = None

    

    def _register_page(self, page: Page) -> None:
        """Context ``page`` event handler — assign an id and auto-activate."""
        pid = self._next_id
        self._next_id += 1
        self._pages[pid] = page
        self._active_id = pid  
        page.on("close", partial(self._forget_page, pid))

    def _forget_page(self, pid: int, *_: object) -> None:
        self._pages.pop(pid, None)

    @property
    def page(self) -> Page:
        """The active tab. Falls back to the highest open tab id."""
        page = self._pages.get(self._active_id)
        if page is not None and not page.is_closed():
            return page
        for pid in sorted(self._pages, reverse=True):
            if not self._pages[pid].is_closed():
                self._active_id = pid
                return self._pages[pid]
        raise RuntimeError("No open tabs — browser not started or all closed.")

    def tabs(self) -> list[dict]:
        """Return ``[{id, url, active}]`` for every open tab (sorted by id)."""
        out: list[dict] = []
        for pid in sorted(self._pages):
            pg = self._pages[pid]
            if pg.is_closed():
                continue
            out.append(
                {"id": pid, "url": pg.url, "active": pid == self._active_id}
            )
        return out

    async def switch_tab(self, tab_id: int) -> None:
        """Make *tab_id* the active tab and bring it to the foreground."""
        pg = self._pages.get(tab_id)
        if pg is None or pg.is_closed():
            raise ValueError(f"switch_tab: tab {tab_id} does not exist")
        self._active_id = tab_id
        await pg.bring_to_front()

    async def close_tab(self, tab_id: int) -> None:
        """Close *tab_id*; activate the next remaining tab if it was active."""
        pg = self._pages.get(tab_id)
        if pg is None:
            raise ValueError(f"close_tab: tab {tab_id} does not exist")
        if len([p for p in self._pages.values() if not p.is_closed()]) <= 1:
            raise ValueError("close_tab: refusing to close the only open tab")
        await pg.close()
        self._pages.pop(tab_id, None)
        if self._active_id == tab_id and self._pages:
            self._active_id = max(self._pages)

    

    async def navigate(self, url: str) -> None:
        """Navigate the active tab to *url* and wait for the DOM."""
        if not url.startswith("http"):
            url = "https://" + url
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    def current_url(self) -> str:
        """Return the URL of the active tab."""
        try:
            return self.page.url
        except RuntimeError:
            return ""

    

    async def start_screencast(
        self,
        emit_frame: FrameCallback,
        *,
        quality: int = 60,
        every_nth_frame: int = 1,
    ) -> CDPSession:
        """Begin streaming JPEG frames from the active page via CDP.

        *emit_frame* receives a base64-encoded JPEG string each time the
        compositor produces a new frame.  The CDP session is returned so the
        caller can later call ``stop_screencast``.
        """
        page = self.page
        cdp: CDPSession = await page.context.new_cdp_session(page)
        loop = asyncio.get_running_loop()

        def _on_frame(event: dict[str, Any]) -> None:
            emit_frame(event["data"])
            asyncio.run_coroutine_threadsafe(
                cdp.send(
                    "Page.screencastFrameAck",
                    {"sessionId": event["sessionId"]},
                ),
                loop,
            )

        cdp.on("Page.screencastFrame", _on_frame)
        await cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": quality,
                "everyNthFrame": every_nth_frame,
                "maxWidth": config.viewport_width,
                "maxHeight": config.viewport_height,
            },
        )
        self._cdp_session = cdp
        return cdp

    async def stop_screencast(self) -> None:
        """Stop the running screencast, if any."""
        cdp = getattr(self, "_cdp_session", None)
        if cdp is None:
            return
        try:
            await cdp.send("Page.stopScreencast")
            await cdp.detach()
        except Exception:
            pass
        self._cdp_session = None
