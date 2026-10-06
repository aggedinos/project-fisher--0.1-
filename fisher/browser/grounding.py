"""Resolve observed element IDs to resilient Playwright locators."""

from __future__ import annotations

import json

from playwright.async_api import Locator, Page

from fisher.browser.models import GroundedTarget, StaleElementError


class ElementGrounder:
    def __init__(self) -> None:
        self._targets: dict[str, GroundedTarget] = {}
        self._page: Page | None = None

    def update(self, page: Page, targets: dict[str, GroundedTarget]) -> None:
        self._page = page
        self._targets = targets

    def clear(self) -> None:
        self._page = None
        self._targets = {}

    def info(self, element_id: str) -> GroundedTarget:
        try:
            return self._targets[element_id]
        except KeyError as exc:
            raise StaleElementError(
                f"Element {element_id!r} is not in the latest observation; observe again"
            ) from exc

    async def locator(self, element_id: str) -> Locator:
        target = self.info(element_id)
        page = self._page
        if page is None or page.is_closed():
            raise StaleElementError("The observed page has closed")
        marker = page.locator(f"[data-fisher-id={json.dumps(element_id)}]")
        try:
            if await marker.count() != 1 or not await marker.is_visible():
                raise StaleElementError(f"Element {element_id!r} is stale or hidden")
        except StaleElementError:
            raise
        except Exception as exc:
            raise StaleElementError(f"Element {element_id!r} is stale") from exc

        info = target.info
        candidates: list[Locator] = []
        if info.role and info.name:
            candidates.append(page.get_by_role(info.role, name=info.name, exact=True))
        if info.name and info.role in {"textbox", "combobox", "checkbox", "radio"}:
            candidates.append(page.get_by_label(info.name, exact=True))
        if target.test_id:
            candidates.append(page.get_by_test_id(target.test_id))
        if target.dom_id:
            candidates.append(page.locator(f"[id={json.dumps(target.dom_id)}]"))
        if info.name and info.role in {"button", "link"}:
            candidates.append(page.get_by_text(info.name, exact=True))
        candidates.append(marker)

        # A semantic locator is used only when it uniquely identifies the
        # *same* observed element. This avoids a duplicate-label misclick.
        for candidate in candidates:
            try:
                if (
                    await candidate.count() == 1
                    and await candidate.get_attribute("data-fisher-id") == element_id
                ):
                    return candidate
            except Exception:
                continue
        raise StaleElementError(f"Element {element_id!r} can no longer be grounded")
