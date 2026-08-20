"""Modern desktop GUI for the browser automation agent (pywebview).

The interface is HTML/CSS/JS rendered in a native window via the OS webview
(Edge WebView2 on Windows 11) — no browser bundling needed for the UI. The
Python side keeps the original architecture: the agent runs on a background
thread with its own asyncio loop, and progress is pushed to the page through
``window.evaluate_js``. Stop is cooperative — the agent finishes the current
step, closes the browser, and exits cleanly.

Falls back to the legacy CustomTkinter GUI if ``pywebview`` is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading

from dotenv import load_dotenv

import paths  
import settings_manager
from agent import BrowserAgent
from config import config
from llm import NvidiaProvider, OllamaProvider, build_provider

APP_TITLE = "Browser Automation Agent"
DEFAULT_URL = "https://www.google.com"

SUGGESTED_OLLAMA_MODELS = [
    "llama3.2-vision",
    "llava",
    "llava:13b",
    "qwen2.5vl",
    "moondream",
]
SUGGESTED_NVIDIA_MODELS = [
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3.5-vision-instruct",
    "meta/llama-3.1-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
]


class Api:
    """Bridge exposed to the page as ``window.pywebview.api``.

    Methods invoked from JS return immediately; long-running work happens on
    daemon threads and streams results back to the page via ``_emit``.
    """

    def __init__(self) -> None:
        self._window = None  
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None
        self._agent: BrowserAgent | None = None
        self._agent_loop: asyncio.AbstractEventLoop | None = None

        self._apply_settings_to_config(settings_manager.load_settings())

    @staticmethod
    def _apply_settings_to_config(settings: dict) -> None:
        """Push local settings onto the live ``config`` singleton."""
        config.use_real_chrome = bool(settings.get("use_real_chrome", config.use_real_chrome))
        config.headless = bool(settings.get("headless", config.headless))

    

    def _emit(self, **event: object) -> None:
        """Push a JSON event to ``window.__agentEvent`` on the page."""
        win = self._window
        if win is None:
            return
        payload = json.dumps(event)
        try:
            win.evaluate_js(f"window.__agentEvent({json.dumps(payload)})")
        except Exception:  
            pass

    

    def get_local_settings(self) -> dict:
        """Return local-only settings from disk for the frontend to apply on load."""
        return settings_manager.load_settings()

    def save_local_setting(self, key: str, value: object) -> dict:
        """Persist one local setting to disk and apply it to the live config."""
        try:
            settings = settings_manager.update_setting(key, value)
        except KeyError:
            return settings_manager.load_settings()
        self._apply_settings_to_config(settings)
        return settings

    def get_settings(self) -> dict:
        """Legacy bridge — returns local settings for backward compat."""
        return self.get_local_settings()

    def save_setting(self, key: str, value: object) -> dict:
        """Legacy bridge — routes to save_local_setting."""
        return self.save_local_setting(key, value)

    

    def get_state(self) -> dict:
        return {
            "provider": config.provider,
            "click_mode": config.click_mode,
            "use_real_chrome": config.use_real_chrome,
            "thinking_level": config.thinking_level,
            "gemini_model": "flash" if "flash" in config.gemini_model else "pro",
            "ollama_base_url": config.ollama_base_url,
            "ollama_model": config.ollama_model,
            "ollama_models": SUGGESTED_OLLAMA_MODELS,
            "nvidia_model": config.nvidia_model,
            "nvidia_api_key": os.getenv("NVIDIA_API_KEY", "").strip(),
            "nvidia_models": SUGGESTED_NVIDIA_MODELS,
            "headless": config.headless,
            "max_steps": config.max_steps,
            "custom_instructions": config.system_instructions,
        }

    

    def refresh_ollama_models(self, base_url: str) -> None:
        base = (base_url or "").strip() or config.ollama_base_url

        def work() -> None:
            try:
                models = OllamaProvider.list_models(base)
                self._emit(
                    kind="ollama-models",
                    models=models or SUGGESTED_OLLAMA_MODELS,
                    error=None if models else "Ollama has no models installed.",
                )
            except Exception as exc:  
                self._emit(
                    kind="ollama-models",
                    models=[],
                    error=f"Could not reach Ollama at {base}: {exc}",
                )

        threading.Thread(target=work, daemon=True).start()

    def refresh_nvidia_models(self, api_key: str) -> None:
        key = (api_key or "").strip()

        def work() -> None:
            try:
                models = NvidiaProvider.list_models(key, config.nvidia_base_url)
                self._emit(
                    kind="nvidia-models",
                    models=models or SUGGESTED_NVIDIA_MODELS,
                    error=None if models else "No NVIDIA models for this key.",
                )
            except Exception as exc:  
                self._emit(
                    kind="nvidia-models",
                    models=[],
                    error=f"NVIDIA model list failed: {exc}",
                )

        threading.Thread(target=work, daemon=True).start()

    

    def run_agent(self, prompt: str, options: dict) -> None:
        if self._worker and self._worker.is_alive():
            return

        task = str(prompt or "").strip()
        if not task:
            self._emit(kind="status", text="Enter a task first", tone="err")
            return

        url = str(options.get("url", "")).strip() or DEFAULT_URL
        provider_name = str(options.get("provider", config.provider)).lower().strip()
        click_mode = (
            "text" if str(options.get("click_mode")) == "text" else "vision"
        )
        config.use_real_chrome = bool(
            options.get("use_real_chrome", options.get("real_chrome", config.use_real_chrome))
        )

        
        
        
        config.headless = bool(options.get("headless", config.headless))
        config.max_steps = int(options.get("max_steps", 30))
        config.system_instructions = str(options.get("custom_instructions", "")).strip()

        if provider_name == "nvidia":
            send_images = bool(options.get("nvidia_send_images", True))
        elif provider_name == "ollama":
            send_images = bool(options.get("ollama_send_images", True))
        else:
            send_images = True

        effort = str(options.get("effort", "fast")).strip().lower()
        gemini_model_mapping = {
            "fast": "gemini-2.5-flash",
            "thinking": "gemini-2.5-pro",
            "pro": "gemini-3.1-pro-preview",
            "flash": "gemini-2.5-flash",
        }
        gemini_model_name = gemini_model_mapping.get(effort, "gemini-2.5-flash")

        try:
            provider = build_provider(
                provider=provider_name,
                temperature=config.temperature,
                gemini_model=gemini_model_name,
                ollama_base_url=str(options.get("ollama_base_url", "")).strip()
                or config.ollama_base_url,
                ollama_model=str(options.get("ollama_model", "")).strip()
                or config.ollama_model,
                nvidia_api_key=str(options.get("nvidia_api_key", "")).strip(),
                nvidia_model=str(options.get("nvidia_model", "")).strip()
                or config.nvidia_model,
                nvidia_base_url=config.nvidia_base_url,
                send_images=send_images,
            )
        except ValueError as exc:
            self._emit(kind="done", text=str(exc), ok=False)
            return

        self._stop_event.clear()
        self._emit(kind="log", text=f"Provider : {provider.label} ({provider.model})")
        self._emit(kind="log", text=f"Clicking : {click_mode} mode")
        self._emit(
            kind="log",
            text="Browser  : "
            + ("real Chrome profile" if config.use_real_chrome else "Chromium"),
        )
        self._emit(kind="log", text=f"Task     : {task}")
        self._emit(kind="log", text=f"Start    : {url}\n")

        def work() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._agent_loop = loop
            try:
                agent = BrowserAgent(provider=provider)
                self._agent = agent

                def log_and_emit(line: str) -> None:
                    self._emit(kind="log", text=str(line))

                def emit_frame(b64_jpeg: str) -> None:
                    self._emit(kind="screencast_frame", data=b64_jpeg)

                result = loop.run_until_complete(
                    agent.run(
                        task=task,
                        start_url=url,
                        log=log_and_emit,
                        should_stop=self._stop_event.is_set,
                        click_mode=click_mode,
                        system_instructions=config.system_instructions,
                        emit_frame=emit_frame,
                    )
                )

                ok = (
                    not str(result).startswith("[ERROR]")
                    and str(result) != "Stopped by user."
                )
                self._emit(kind="done", text=str(result), ok=ok)
            except Exception as exc:  
                self._emit(kind="done", text=f"[ERROR] {exc}", ok=False)
            finally:
                self._agent = None
                self._agent_loop = None
                loop.close()

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def stop_agent(self) -> None:
        self._stop_event.set()

    def warm_up_chrome_profile(self) -> None:
        """Open the persistent Chrome profile headed for a one-time login.

        The user signs into Google (and anything else they want the agent to
        inherit) in a normal visible window, then closes it. Cookies persist
        in ``paths.chrome_profile_dir()`` so subsequent headless agent runs
        look like an existing logged-in user — no more 'unusual traffic'.
        """
        if self._worker and self._worker.is_alive():
            self._emit(
                kind="warmup_done",
                ok=False,
                error="An agent is running — stop it before warming up.",
            )
            return
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._emit(
                kind="warmup_done",
                ok=False,
                error="Warm-up is already in progress.",
            )
            return

        def work() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._warm_up_async())
                self._emit(kind="warmup_done", ok=True)
            except Exception as exc:  
                self._emit(kind="warmup_done", ok=False, error=str(exc))
            finally:
                loop.close()

        self._warmup_thread = threading.Thread(target=work, daemon=True)
        self._warmup_thread.start()

    async def _warm_up_async(self) -> None:
        """Coroutine body of warm_up_chrome_profile — see its docstring."""
        from browser import _close_chrome
        from playwright.async_api import async_playwright

        profile_path = paths.chrome_profile_dir()
        self._emit(
            kind="log",
            text="[WARMUP] Closing any open Chrome to release the profile lock...",
        )
        _close_chrome()
        self._emit(
            kind="log",
            text=f"[WARMUP] Opening headed Chrome at {profile_path}",
        )
        self._emit(
            kind="log",
            text="[WARMUP] Sign into Google in the window that appears, then close it when done.",
        )

        async with async_playwright() as pw:
            context = await pw.chromium.launch_persistent_context(
                profile_path,
                channel="chrome",
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-session-crashed-bubble",
                    "--hide-crash-restore-bubble",
                ],
                ignore_default_args=["--enable-automation"],
            )

            closed = asyncio.Event()
            context.on("close", lambda: closed.set())

            page = context.pages[0] if context.pages else await context.new_page()
            try:
                await page.goto(
                    "https://www.google.com/",
                    wait_until="domcontentloaded",
                    timeout=20_000,
                )
            except Exception:
                pass

            await closed.wait()

        self._emit(
            kind="log",
            text="[WARMUP] Chrome closed — cookies saved. Future runs reuse this session.",
        )

    def inject_manual_click(self, x: int, y: int) -> None:
        """Forward a user click from the canvas to the live Playwright page."""
        agent = self._agent
        loop = self._agent_loop
        if agent is None or loop is None:
            return
        browser = getattr(agent, "_live_browser", None)
        if browser is None:
            return

        async def _click() -> None:
            try:
                await browser.page.mouse.click(int(x), int(y))
            except Exception:
                pass

        asyncio.run_coroutine_threadsafe(_click(), loop)

    def shutdown(self) -> None:
        self._stop_event.set()


def _ui_html_path() -> str:
    """Return the path to the built single-file UI."""
    return str(paths.bundle_dir() / "web" / "index.html")


def launch() -> None:
    """Build and run the modern desktop app."""
    load_dotenv(paths.env_file())

    try:
        import webview
    except ImportError:
        from gui import launch as legacy_launch

        legacy_launch()
        return

    html_path = _ui_html_path()
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            f"UI not built. Run 'npm run build' in the project directory first.\n"
            f"  Expected: {html_path}"
        )

    api = Api()
    window = webview.create_window(
        APP_TITLE,
        url=html_path,
        js_api=api,
        width=1280,
        height=860,
        min_size=(1024, 760),
        background_color="#0b0f1a",
        text_select=True,
    )
    api._window = window
    window.events.closed += api.shutdown
    webview.start()


if __name__ == "__main__":
    launch()
