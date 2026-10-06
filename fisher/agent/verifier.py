"""Action-specific verification from observations before and after a tool call."""

from __future__ import annotations

from fisher.models import Observation, ToolCall

from .memory import safe_url


_OBSERVATIONAL = {"read_page", "screenshot"}
_INPUT_ACTIONS = {"fill_element", "type_text"}
_CLICK_ACTIONS = {"click_element", "click_coordinates"}


def _element_states(observation: Observation) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.role, item.name, item.text, item.visible, item.enabled,
         item.href, item.value if item.input_type != "password" else "[password]")
        for item in observation.elements
    )


def verify_change(
    call: ToolCall, before: Observation, after: Observation
) -> tuple[bool, list[str]]:
    """Return whether an action has observable evidence, with the signals found.

    The boolean means *verified*, not necessarily changed: read-only tools can
    succeed without changing a page. The registry separately sets ``changed``.
    """
    evidence: list[str] = []
    if before.url != after.url:
        evidence.append(f"URL changed: {safe_url(before.url)} -> {safe_url(after.url)}")
    if before.active_tab_id != after.active_tab_id:
        evidence.append("active tab changed")
    if [(t.id, t.url) for t in before.tabs] != [(t.id, t.url) for t in after.tabs]:
        evidence.append("tab set or tab URL changed")
    if before.title != after.title:
        evidence.append("page title changed")
    if before.text != after.text:
        evidence.append("page text changed")
    if _element_states(before) != _element_states(after):
        evidence.append("interactive element state changed")
    if before.fingerprint and after.fingerprint and before.fingerprint != after.fingerprint:
        evidence.append("page fingerprint changed")

    name = call.name
    if name in _OBSERVATIONAL:
        if after.url or after.text or after.elements:
            return True, evidence or ["page observation captured"]
        return False, ["page observation is empty"]
    if name == "navigate":
        relevant = [signal for signal in evidence if signal.startswith("URL changed")
                    or signal in {"page text changed", "page title changed",
                                  "page fingerprint changed", "tab set or tab URL changed"}]
        return bool(relevant), relevant or ["navigation produced no observable page change"]
    if name == "switch_tab":
        relevant = [signal for signal in evidence if signal == "active tab changed"]
        return bool(relevant), relevant or ["active tab did not change"]
    if name == "close_tab":
        relevant = [signal for signal in evidence if signal == "tab set or tab URL changed"]
        return bool(relevant), relevant or ["tab set did not change"]
    if name in _INPUT_ACTIONS:
        relevant = [signal for signal in evidence if signal in {
            "interactive element state changed", "page fingerprint changed", "page text changed"
        }]
        return bool(relevant), relevant or ["input value and page state did not change"]
    if name == "scroll":
        relevant = [signal for signal in evidence if signal in {
            "page fingerprint changed", "page text changed", "interactive element state changed"
        }]
        return bool(relevant), relevant or ["scroll position or visible content did not change"]
    if name in _CLICK_ACTIONS or name == "press_key":
        return bool(evidence), evidence or ["interaction produced no observable page change"]
    return bool(evidence), evidence or ["no observable state change"]
