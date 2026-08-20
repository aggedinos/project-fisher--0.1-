"""Action definitions and execution against the browser controller."""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page

from browser import BrowserController
from config import config


async def _random_delay() -> None:
    """Sleep for a random duration between min/max_action_delay_ms."""
    delay = random.uniform(config.min_action_delay_ms, config.max_action_delay_ms)
    await asyncio.sleep(delay / 1000)


@dataclass
class Action:
    """A single browser action produced by the agent."""

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: int = 100

    def __str__(self) -> str:
        return f"{self.action_type} {self.params}"


async def _find_and_click(page: Page, text_to_find: str) -> None:
    """Click the first interactive element whose text contains *text_to_find*."""
    clicked: bool = await page.evaluate(
        """(searchText) => {
            const lower = searchText.toLowerCase().trim();
            const tags = 'button, a, [role="button"], '
                       + 'input[type="button"], input[type="submit"], label';
            for (const el of document.querySelectorAll(tags)) {
                const elText = (el.textContent || el.innerText || el.value || '')
                                .trim().toLowerCase();
                if (elText.includes(lower)) { el.click(); return true; }
            }
            return false;
        }""",
        text_to_find,
    )
    if not clicked:
        try:
            await page.locator(f"text={text_to_find}").first.click(timeout=3000)
        except Exception as exc:
            raise ValueError(
                f"find_and_click: no element found with text '{text_to_find}'"
            ) from exc


async def _element_text_at_point(page: Page, x: int, y: int) -> str | None:
    """Best-effort visible text/label of whatever sits at pixel (x, y).

    Lets a coordinate click try Playwright's resilient text locator first,
    instead of trusting a raw pixel that can drift once the layout shifts or
    content lazily loads. Returns None for empty text or text long enough
    that a substring match could land on the wrong element.
    """
    try:
        text: str | None = await page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (!el) return null;
                const t = (el.innerText || el.textContent
                           || el.getAttribute('aria-label')
                           || el.getAttribute('title') || '').trim();
                return t || null;
            }""",
            [x, y],
        )
    except Exception:
        return None
    if text and 0 < len(text) <= 80:
        return text
    return None


async def execute_action(browser: BrowserController, action: Action) -> str | None:
    """Execute *action* against *browser*.

    Returns a non-None string only for the ``done`` action (task complete);
    all other actions return None. Raises ``ValueError`` for unknown actions
    or invalid parameters so the agent's retry logic can react.
    """
    t = action.action_type
    p = action.params
    page = browser.page

    if t == "click":
        if "x" in p and "y" in p and p.get("x") is not None and p.get("y") is not None:
            x, y = int(p["x"]), int(p["y"])
            if not (0 <= x <= config.viewport_width) or not (
                0 <= y <= config.viewport_height
            ):
                raise ValueError(
                    f"click coordinates ({x},{y}) out of bounds — must be "
                    f"0..{config.viewport_width} x, 0..{config.viewport_height} y"
                )

            
            
            text_hint = p.get("text") or await _element_text_at_point(page, x, y)
            clicked_via_text = False
            if text_hint:
                try:
                    await _find_and_click(page, text_hint)
                    clicked_via_text = True
                except Exception:
                    clicked_via_text = False

            if not clicked_via_text:
                
                
                await page.wait_for_timeout(200)
                await page.mouse.click(x, y)
        elif p.get("text"):
            await _find_and_click(page, p["text"])
        else:
            raise ValueError("click requires x,y coordinates (or a 'text' fallback)")
        await _random_delay()

    elif t == "find_and_click":
        await _find_and_click(page, p.get("text", ""))
        await _random_delay()

    elif t == "type":
        await page.keyboard.type(p.get("text", ""))
        await _random_delay()

    elif t == "key":
        await page.keyboard.press(p.get("key", "Enter"))
        await _random_delay()

    elif t == "scroll":
        direction = p.get("direction", "down")
        amount = int(p.get("amount", 300))
        delta_y = amount if direction == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta_y})")
        await _random_delay()

    elif t == "fill":
        await page.locator(p.get("selector", "")).first.fill(p.get("text", ""))
        await _random_delay()

    elif t == "navigate":
        url = p.get("url", "")
        if not url.startswith("http"):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await _random_delay()

    elif t == "switch_tab":
        await browser.switch_tab(int(p.get("tab_id", 0)))
        await _random_delay()

    elif t == "close_tab":
        await browser.close_tab(int(p.get("tab_id", 0)))
        await _random_delay()

    elif t == "done":
        return p.get("result", "Task completed.")

    else:
        raise ValueError(f"Unknown action type: '{t!r}'")

    return None
