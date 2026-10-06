"""Browser tool definitions and their small dispatch layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from fisher.browser.controller import BrowserController
from fisher.models import RiskLevel, ToolSpec
from fisher.tools.models import (
    CoordinatesArgs,
    ElementArgs,
    KeyArgs,
    NavigateArgs,
    ReadPageArgs,
    ScrollArgs,
    TabArgs,
    TextArgs,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments: type[BaseModel]
    risk: RiskLevel = RiskLevel.LOW
    timeout_seconds: float = 15.0
    retryable: bool = True

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.arguments.model_json_schema(),
            risk=self.risk,
            timeout_seconds=self.timeout_seconds,
            retryable=self.retryable,
        )


DEFINITIONS: dict[str, ToolDefinition] = {
    item.name: item
    for item in [
        ToolDefinition(
            "navigate", "Open an absolute HTTP or HTTPS URL.", NavigateArgs, timeout_seconds=35
        ),
        ToolDefinition(
            "read_page", "Read the current compact page observation.", ReadPageArgs, retryable=False
        ),
        ToolDefinition(
            "click_element", "Click an element ID from the latest observation.", ElementArgs
        ),
        ToolDefinition("fill_element", "Replace a field's value with text.", TextArgs),
        ToolDefinition("type_text", "Type text into an observed field.", TextArgs),
        ToolDefinition("press_key", "Press a supported browser key.", KeyArgs),
        ToolDefinition("scroll", "Scroll the active page by signed pixels.", ScrollArgs),
        ToolDefinition("switch_tab", "Activate an observed browser tab.", TabArgs),
        ToolDefinition("close_tab", "Close a tab when another tab remains.", TabArgs),
        ToolDefinition(
            "click_coordinates",
            "Click viewport coordinates only when no semantic target exists.",
            CoordinatesArgs,
            risk=RiskLevel.HIGH,
        ),
    ]
}


async def invoke_browser_tool(
    browser: BrowserController, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if name == "navigate":
        await browser.navigate(arguments["url"])
    elif name == "read_page":
        return {}
    elif name == "click_element":
        await browser.click_element(arguments["element_id"])
    elif name == "fill_element":
        await browser.fill_element(arguments["element_id"], arguments["text"])
    elif name == "type_text":
        await browser.type_text(arguments["element_id"], arguments["text"])
    elif name == "press_key":
        await browser.press_key(arguments["key"])
    elif name == "scroll":
        await browser.scroll(arguments["delta_y"])
    elif name == "switch_tab":
        await browser.switch_tab(arguments["tab_id"])
    elif name == "close_tab":
        await browser.close_tab(arguments["tab_id"])
    elif name == "click_coordinates":
        await browser.click_coordinates(arguments["x"], arguments["y"])
    else:
        raise ValueError(f"Unknown tool {name!r}")
    return {}
