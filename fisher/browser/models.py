"""Internal browser state used to ground a short-lived element ID."""

from __future__ import annotations

from dataclasses import dataclass

from fisher.models import ElementInfo


class BrowserNotStarted(RuntimeError):
    pass


class StaleElementError(ValueError):
    """An element from the latest observation has disappeared or changed."""


@dataclass(frozen=True)
class GroundedTarget:
    info: ElementInfo
    dom_id: str = ""
    test_id: str = ""
    name_attr: str = ""


@dataclass(frozen=True)
class PerceivedPage:
    title: str
    text: str
    targets: dict[str, GroundedTarget]
    scroll_x: int
    scroll_y: int
