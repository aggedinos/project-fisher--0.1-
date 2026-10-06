"""A browser owned by Fisher, with tab tracking and short-lived grounding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Page,
    Playwright,
    Request,
    Route,
    async_playwright,
)

from fisher.browser.grounding import ElementGrounder
from fisher.browser.models import BrowserNotStarted
from fisher.browser.perception import inspect_page
from fisher.config.paths import default_data_dir, ensure_directory, profile_dir
from fisher.models import Observation, TabInfo
from fisher.security.policy import SecurityError, SecurityPolicy


class BrowserController:
    def __init__(
        self,
        *,
        allow_private_network: bool = False,
        policy: SecurityPolicy | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.policy = policy or SecurityPolicy(allow_private_network=allow_private_network)
        self.data_dir = Path(data_dir or os.environ.get("FISHER_DATA_DIR") or default_data_dir())
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[str, Page] = {}
        self._active_tab_id = ""
        self._next_tab = 1
        self._observation_number = 0
        self._last_observation: Observation | None = None
        self._tab_titles: dict[str, str] = {}
        self._grounder = ElementGrounder()
        self._cdp: CDPSession | None = None
        self._screencast_callback: Callable[[str], None] | None = None
        self._page_registered = asyncio.Event()

    @property
    def page(self) -> Page:
        page = self._pages.get(self._active_tab_id)
        if page is not None and not page.is_closed():
            return page
        for tab_id, candidate in reversed(list(self._pages.items())):
            if not candidate.is_closed():
                self._active_tab_id = tab_id
                return candidate
        raise BrowserNotStarted("No open browser tab")

    @property
    def last_observation(self) -> Observation | None:
        return self._last_observation

    async def start(
        self,
        profile: str = "temporary",
        headless: bool = True,
        executable_path: str | None = None,
    ) -> None:
        if self._playwright is not None:
            raise RuntimeError("Browser has already started")
        if profile not in {"temporary", "persistent"}:
            raise ValueError("profile must be 'temporary' or 'persistent'")
        executable_path = executable_path or os.environ.get("FISHER_CHROMIUM_EXECUTABLE") or None
        self._playwright = await async_playwright().start()
        try:
            launch_options: dict[str, Any] = {"headless": headless}
            if executable_path:
                launch_options["executable_path"] = str(executable_path)
            common_context: dict[str, Any] = {
                "viewport": {"width": 1280, "height": 720},
                "accept_downloads": False,
                "service_workers": "block",
            }
            if profile == "persistent":
                persistent_path = ensure_directory(profile_dir(self.data_dir))
                try:
                    self._context = await self._playwright.chromium.launch_persistent_context(
                        str(persistent_path), **launch_options, **common_context
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot open Fisher browser profile at {persistent_path}. "
                        "Close any Fisher browser using this profile and retry."
                    ) from exc
            else:
                self._browser = await self._playwright.chromium.launch(**launch_options)
                self._context = await self._browser.new_context(**common_context)

            self._context.on("page", self._register_page)
            await self._context.route("**/*", self._guard_route)
            for existing in self._context.pages:
                self._register_page(existing)
            if not self._pages:
                self._register_page(await self._context.new_page())
        except Exception:
            await self.close()
            raise

    async def _guard_route(self, route: Route, request: Request) -> None:
        try:
            scheme = urlsplit(request.url).scheme.lower()
            if scheme in {"data", "blob"} and not request.is_navigation_request():
                await route.continue_()
                return
            await self.policy.validate_url_destination(request.url)
        except SecurityError:
            await route.abort("blockedbyclient")
        else:
            await route.continue_()

    def _register_page(self, page: Page) -> None:
        for tab_id, known in self._pages.items():
            if known is page:
                return
        tab_id = f"t{self._next_tab}"
        self._next_tab += 1
        self._pages[tab_id] = page
        self._active_tab_id = tab_id
        self._page_registered.set()
        self._grounder.clear()
        self._last_observation = None
        page.on("close", lambda: self._forget_page(tab_id))
        page.on("download", lambda download: asyncio.create_task(download.cancel()))
        if self._screencast_callback:
            asyncio.create_task(self._restart_screencast())

    def _forget_page(self, tab_id: str) -> None:
        self._pages.pop(tab_id, None)
        self._tab_titles.pop(tab_id, None)
        if self._active_tab_id == tab_id:
            self._active_tab_id = next(reversed(self._pages), "")
            self._grounder.clear()
            self._last_observation = None
            if self._screencast_callback and self._pages:
                asyncio.create_task(self._restart_screencast())

    def tabs(self) -> list[TabInfo]:
        return [
            TabInfo(
                id=tab_id,
                url=page.url,
                title=self._tab_titles.get(tab_id, ""),
                active=tab_id == self._active_tab_id,
            )
            for tab_id, page in self._pages.items()
            if not page.is_closed()
        ]

    async def observe(self) -> Observation:
        page = self.page
        self._observation_number += 1
        perceived = await inspect_page(page, self._observation_number)
        self._grounder.update(page, perceived.targets)
        self._tab_titles[self._active_tab_id] = perceived.title
        tabs = self.tabs()
        elements = [target.info for target in perceived.targets.values()]
        state = {
            "url": page.url,
            "title": perceived.title,
            "text": perceived.text,
            "elements": [item.model_dump(exclude={"id"}) for item in elements],
            "tabs": [tab.model_dump() for tab in tabs],
            "scroll": [perceived.scroll_x, perceived.scroll_y],
        }
        fingerprint = hashlib.sha256(
            json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self._last_observation = Observation(
            url=page.url,
            title=perceived.title,
            text=perceived.text,
            elements=elements,
            tabs=tabs,
            active_tab_id=self._active_tab_id,
            fingerprint=fingerprint,
        )
        return self._last_observation

    async def screenshot(self) -> bytes:
        """Return a compact JPEG with observed sensitive inputs covered."""
        page = self.page
        mask = [
            page.locator(
                "input[type='password'], input[autocomplete^='cc-'], "
                "input[autocomplete='one-time-code']"
            )
        ]
        if self._last_observation is not None:
            mask.extend(
                page.locator(f'[data-fisher-id="{item.id}"]')
                for item in self._last_observation.elements
                if item.sensitive
            )
        return await page.screenshot(
            type="jpeg",
            quality=70,
            full_page=False,
            mask=mask,
            mask_color="#101820",
        )

    async def navigate(self, url: str) -> None:
        await self.policy.validate_url_destination(url)
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self._grounder.clear()
        self._last_observation = None

    async def click_element(self, element_id: str) -> None:
        target = self._grounder.info(element_id)
        if target.info.href:
            await self.policy.validate_url_destination(target.info.href)
        locator = await self._grounder.locator(element_id)
        opens_tab = await locator.get_attribute("target") == "_blank"
        self._page_registered.clear()
        tab_count = len(self._pages)
        await locator.click(timeout=10_000)
        if len(self._pages) == tab_count:
            try:
                await asyncio.wait_for(
                    self._page_registered.wait(), timeout=1.0 if opens_tab else 0.15
                )
            except asyncio.TimeoutError:
                pass
        self._grounder.clear()
        self._last_observation = None

    async def fill_element(self, element_id: str, text: str) -> None:
        locator = await self._grounder.locator(element_id)
        await locator.fill(text, timeout=10_000)
        self._grounder.clear()
        self._last_observation = None

    async def type_text(self, element_id: str, text: str) -> None:
        locator = await self._grounder.locator(element_id)
        await locator.press_sequentially(text, delay=20, timeout=10_000)
        self._grounder.clear()
        self._last_observation = None

    async def press_key(self, key: str) -> None:
        await self.page.keyboard.press(key)
        self._grounder.clear()
        self._last_observation = None

    async def scroll(self, delta_y: int) -> None:
        await self.page.mouse.wheel(0, delta_y)
        await asyncio.sleep(0.08)
        self._grounder.clear()
        self._last_observation = None

    async def switch_tab(self, tab_id: str) -> None:
        page = self._pages.get(tab_id)
        if page is None or page.is_closed():
            raise ValueError(f"Tab {tab_id!r} does not exist")
        self._active_tab_id = tab_id
        await page.bring_to_front()
        self._grounder.clear()
        self._last_observation = None
        if self._screencast_callback:
            await self._restart_screencast()

    async def close_tab(self, tab_id: str) -> None:
        page = self._pages.get(tab_id)
        if page is None or page.is_closed():
            raise ValueError(f"Tab {tab_id!r} does not exist")
        if len(self.tabs()) <= 1:
            raise ValueError("Cannot close the only open tab")
        await page.close()
        self._forget_page(tab_id)

    async def click_coordinates(self, x: int, y: int) -> None:
        size = self.page.viewport_size or {"width": 1280, "height": 720}
        if not (0 <= x < size["width"] and 0 <= y < size["height"]):
            raise ValueError("Click coordinates are outside the viewport")
        await self.page.mouse.click(x, y)
        self._grounder.clear()
        self._last_observation = None

    async def start_screencast(
        self, emit_frame: Callable[[str], None], *, quality: int = 60
    ) -> None:
        self._screencast_callback = emit_frame
        await self._restart_screencast(quality=quality)

    async def _restart_screencast(self, *, quality: int = 60) -> None:
        await self._detach_screencast()
        if not self._screencast_callback or not self._context:
            return
        cdp = await self._context.new_cdp_session(self.page)

        def on_frame(event: dict[str, Any]) -> None:
            callback = self._screencast_callback
            if callback:
                callback(event["data"])
            asyncio.create_task(
                cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
            )

        cdp.on("Page.screencastFrame", on_frame)
        await cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": max(1, min(100, quality)), "everyNthFrame": 1},
        )
        self._cdp = cdp

    async def _detach_screencast(self) -> None:
        cdp, self._cdp = self._cdp, None
        if cdp is None:
            return
        try:
            await cdp.send("Page.stopScreencast")
            await cdp.detach()
        except Exception:
            pass  # The tab may already have closed.

    async def stop_screencast(self) -> None:
        self._screencast_callback = None
        await self._detach_screencast()

    async def close(self) -> None:
        await self.stop_screencast()
        for obj in (self._context, self._browser, self._playwright):
            if obj is not None:
                try:
                    await (obj.stop() if obj is self._playwright else obj.close())
                except Exception:
                    pass  # Best effort teardown after browser disconnect.
        self._pages.clear()
        self._tab_titles.clear()
        self._active_tab_id = ""
        self._grounder.clear()
        self._last_observation = None
        self._context = None
        self._browser = None
        self._playwright = None

    async def stop(self) -> None:
        await self.close()
