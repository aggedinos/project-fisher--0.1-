"""Typed tool and security boundary tests without network dependencies."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fisher.browser.controller import BrowserController
from fisher.models import (
    ElementInfo,
    Observation,
    PermissionMode,
    RiskLevel,
    ToolCall,
)
from fisher.security.permissions import permitted, requires_approval
from fisher.security.policy import SecurityError, SecurityPolicy
from fisher.tools.registry import ToolRegistry
from fisher.tools.registry import _safe_call, _safe_error


def _page() -> Observation:
    return Observation(
        url="https://example.com/order",
        elements=[
            ElementInfo(id="o1-e1", role="button", name="Place order", input_type="submit"),
            ElementInfo(id="o1-e2", role="button", name="Search"),
            ElementInfo(id="o1-e3", role="textbox", name="Password", input_type="password"),
            ElementInfo(id="o1-e4", role="textbox", name="Card number", sensitive=True),
        ],
    )


class PolicyTests(unittest.TestCase):
    def test_web_urls_only_and_private_network_blocked(self) -> None:
        policy = SecurityPolicy()
        self.assertEqual(policy.validate_url("https://example.com/search?q=x"),
                         "https://example.com/search?q=x")
        for url in (
            "file:///etc/passwd", "javascript:alert(1)", "data:text/html,x",
            "http://localhost:8000/", "http://127.0.0.1/", "http://127.1/",
            "http://10.2.3.4/", "http://[::1]/", "http://169.254.169.254/",
            "http://name:password@example.com/",
        ):
            with self.subTest(url=url), self.assertRaises(SecurityError):
                policy.validate_url(url)

    def test_local_fixture_requires_explicit_override(self) -> None:
        self.assertEqual(
            SecurityPolicy(allow_private_network=True).validate_url("http://127.0.0.1:8080/test"),
            "http://127.0.0.1:8080/test",
        )

    def test_consequential_and_credential_risks(self) -> None:
        policy = SecurityPolicy()
        page = _page()
        self.assertEqual(
            policy.classify(ToolCall(name="click_element", arguments={"element_id": "o1-e1"}), page).level,
            RiskLevel.HIGH,
        )
        self.assertEqual(
            policy.classify(ToolCall(name="click_element", arguments={"element_id": "o1-e2"}), page).level,
            RiskLevel.LOW,
        )
        self.assertEqual(
            policy.classify(ToolCall(name="fill_element", arguments={"element_id": "o1-e3", "text": "secret"}), page).level,
            RiskLevel.HIGH,
        )
        self.assertEqual(
            policy.classify(ToolCall(name="fill_element", arguments={"element_id": "o1-e4", "text": "4111"}), page).level,
            RiskLevel.HIGH,
        )
        self.assertEqual(
            policy.classify(ToolCall(name="press_key", arguments={"key": "Enter"}), page).level,
            RiskLevel.HIGH,
        )
        self.assertEqual(
            policy.classify(ToolCall(name="click_coordinates", arguments={"x": 1, "y": 2}), page).level,
            RiskLevel.HIGH,
        )


    def test_mode_boundaries(self) -> None:
        self.assertFalse(requires_approval(PermissionMode.AUTO, RiskLevel.MEDIUM, "navigate"))
        self.assertTrue(requires_approval(PermissionMode.AUTO, RiskLevel.HIGH, "click_element"))
        self.assertTrue(requires_approval(PermissionMode.SAFE, RiskLevel.MEDIUM, "press_key"))
        self.assertTrue(requires_approval(PermissionMode.SUPERVISED, RiskLevel.LOW, "navigate"))
        self.assertFalse(requires_approval(PermissionMode.SUPERVISED, RiskLevel.LOW, "read_page"))

    def test_tool_diagnostics_do_not_echo_url_tokens(self) -> None:
        call = ToolCall(
            name="navigate",
            arguments={"url": "https://user:pass@example.com/path?api_key=secret123#token"},
        )
        safe = _safe_call(call)
        self.assertEqual(safe.arguments["url"], "https://example.com/path")
        error = _safe_error(
            RuntimeError("Page.goto failed at https://example.com/path?token=secret123"), call
        )
        self.assertNotIn("secret123", error)
        self.assertNotIn("token=", error)


class ScreenshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_sensitive_fields_are_masked_for_ui_and_provider(self) -> None:
        class FakePage:
            def __init__(self) -> None:
                self.screenshot = AsyncMock(return_value=b"\xff\xd8\xffimage")

            def is_closed(self) -> bool:
                return False

            def locator(self, selector: str) -> str:
                return selector

        browser = BrowserController()
        page = FakePage()
        browser._pages = {"t1": page}
        browser._active_tab_id = "t1"
        browser._last_observation = _page()
        self.assertEqual(await browser.screenshot(), b"\xff\xd8\xffimage")
        options = page.screenshot.await_args.kwargs
        self.assertIn("input[type='password']", options["mask"][0])
        self.assertIn('[data-fisher-id="o1-e4"]', options["mask"])
        self.assertEqual(options["mask_color"], "#101820")


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    class FakeBrowser:
        policy = SecurityPolicy()
        last_observation = _page()

    async def test_unknown_tool_and_strict_schema_are_rejected(self) -> None:
        registry = ToolRegistry(self.FakeBrowser())
        unknown = await registry.execute(ToolCall(name="run_python", arguments={"source": "print(1)"}))
        self.assertFalse(unknown.success)
        self.assertEqual(unknown.error_code, "unknown_tool")
        invalid = await registry.execute(ToolCall(name="scroll", arguments={"delta_y": "500"}))
        self.assertFalse(invalid.success)
        self.assertEqual(invalid.error_code, "invalid_arguments")
        extra = await registry.execute(ToolCall(name="navigate", arguments={"url": "https://example.com", "script": "x"}))
        self.assertEqual(extra.error_code, "invalid_arguments")

    async def test_policy_and_approval_fail_closed_before_browser_action(self) -> None:
        registry = ToolRegistry(self.FakeBrowser())
        denied_url = await registry.execute(ToolCall(name="navigate", arguments={"url": "http://127.0.0.1/"}))
        self.assertEqual(denied_url.error_code, "blocked_url")
        submit = await registry.execute(ToolCall(name="click_element", arguments={"element_id": "o1-e1"}))
        self.assertEqual(submit.error_code, "permission_required")

    async def test_permission_request_redacts_entered_text(self) -> None:
        requests = []
        allowed = await permitted(
            mode=PermissionMode.SAFE,
            call=ToolCall(name="fill_element", arguments={"element_id": "o1-e3", "text": "secret"}),
            risk=RiskLevel.HIGH,
            reason="credential field",
            approve=lambda request: requests.append(request) or True,
        )
        self.assertTrue(allowed)
        self.assertEqual(requests[0].call.arguments["text"], "[redacted]")
        await permitted(
            mode=PermissionMode.SUPERVISED,
            call=ToolCall(name="navigate", arguments={"url": "https://example.com/a?token=secret123"}),
            risk=RiskLevel.LOW,
            reason="Open a web page",
            approve=lambda request: requests.append(request) or True,
        )
        self.assertEqual(requests[1].call.arguments["url"], "https://example.com/a")

    async def test_dns_resolution_to_private_ip_is_blocked(self) -> None:
        policy = SecurityPolicy()
        record = (2, 1, 6, "", ("10.0.0.8", 0))
        with patch("fisher.security.policy.socket.getaddrinfo", return_value=[record]):
            with self.assertRaises(SecurityError):
                await policy.validate_url_destination("https://public-looking.example/secret")


if __name__ == "__main__":
    unittest.main()
