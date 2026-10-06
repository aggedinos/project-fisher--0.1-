"""End-to-end browser flows against a deterministic local HTTP site.

The tests use stdlib unittest so they can also run through pytest discovery.
"""

from __future__ import annotations

import os
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest import IsolatedAsyncioTestCase

from fisher.agent.orchestrator import AgentOrchestrator
from fisher.browser.controller import BrowserController
from fisher.models import (
    AgentEvent,
    ModelDecision,
    Observation,
    PermissionMode,
    PermissionRequest,
    PlanStep,
    RiskLevel,
    TaskPlan,
    ToolCall,
    ToolResult,
)
from fisher.tools.registry import ToolRegistry


FIXTURES = Path(__file__).parent / "fixtures"
BROWSER_EXECUTABLE = (
    os.environ.get("FISHER_TEST_BROWSER")
    or os.environ.get("FISHER_EXECUTABLE_PATH")
    or next(
        (
            path
            for name in ("chromium", "chromium-browser", "google-chrome", "chrome")
            if (path := shutil.which(name))
        ),
        None,
    )
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def _element(observation: Observation, name: str, role: str | None = None) -> str:
    matching = [
        item.id
        for item in observation.elements
        if item.name == name and (role is None or item.role == role)
    ]
    if len(matching) != 1:
        raise AssertionError(
            f"Expected one element named {name!r} with role {role!r}; "
            f"observed {[(item.role, item.name) for item in observation.elements]!r}"
        )
    return matching[0]


class _ShopProvider:
    label = "fixture-shop"

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        return TaskPlan(
            goal=task,
            steps=[
                PlanStep(id="search", description="Find headphones"),
                PlanStep(id="open", description="Open the matching product"),
                PlanStep(id="report", description="Report the observed price"),
            ],
        )

    async def decide(self, task, plan, observation, memory, tools, image=None) -> ModelDecision:
        if observation.url.endswith("/product.html"):
            return ModelDecision(
                answer=f"Studio headphones cost $79.95 on {observation.url}",
                completed_step_ids=["report"],
            )
        if "One result for headphones" in observation.text:
            return ModelDecision(
                tool_call=ToolCall(
                    name="click_element",
                    arguments={"element_id": _element(observation, "Open Studio headphones")},
                ),
                completed_step_ids=["open"],
            )
        query = next(item for item in observation.elements if item.name == "Product search")
        if query.value != "headphones":
            return ModelDecision(
                tool_call=ToolCall(
                    name="fill_element",
                    arguments={"element_id": query.id, "text": "headphones"},
                )
            )
        return ModelDecision(
            tool_call=ToolCall(
                name="click_element",
                arguments={"element_id": _element(observation, "Search products")},
            ),
            completed_step_ids=["search"],
        )


class _StaleRecoveryProvider:
    label = "fixture-stale"

    def __init__(self) -> None:
        self.old_element_id = ""
        self.attempted_stale = False

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        self.old_element_id = _element(observation, "Continue")
        return TaskPlan(
            goal=task,
            steps=[
                PlanStep(id="refresh", description="Refresh options"),
                PlanStep(id="continue", description="Select the new option"),
            ],
        )

    async def decide(self, task, plan, observation, memory, tools, image=None) -> ModelDecision:
        if "New choice selected" in observation.text:
            return ModelDecision(answer="The new option was selected.")
        if "Options refreshed" not in observation.text:
            return ModelDecision(
                tool_call=ToolCall(
                    name="click_element",
                    arguments={"element_id": _element(observation, "Refresh options")},
                ),
                completed_step_ids=["refresh"],
            )
        if not self.attempted_stale:
            self.attempted_stale = True
            return ModelDecision(
                tool_call=ToolCall(
                    name="click_element", arguments={"element_id": self.old_element_id}
                )
            )
        return ModelDecision(
            tool_call=ToolCall(
                name="click_element",
                arguments={"element_id": _element(observation, "Continue")},
            ),
            completed_step_ids=["continue"],
        )


class _FormProvider:
    label = "fixture-form"

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        return TaskPlan(
            goal=task,
            steps=[
                PlanStep(id="fill", description="Fill the order details"),
                PlanStep(id="submit", description="Submit the order"),
            ],
        )

    async def decide(self, task, plan, observation, memory, tools, image=None) -> ModelDecision:
        for name, value in (
            ("Full name", "Taylor Fisher"),
            ("Email address", "taylor@example.test"),
            ("Delivery note", "Leave at desk"),
        ):
            field = next(item for item in observation.elements if item.name == name)
            if field.value != value:
                return ModelDecision(
                    tool_call=ToolCall(
                        name="fill_element", arguments={"element_id": field.id, "text": value}
                    )
                )
        return ModelDecision(
            tool_call=ToolCall(
                name="click_element",
                arguments={"element_id": _element(observation, "Place order")},
            ),
            completed_step_ids=["fill"],
        )


class _TabProvider:
    label = "fixture-tabs"

    def __init__(self) -> None:
        self.opened = False
        self.read_other = False

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        return TaskPlan(
            goal=task,
            steps=[
                PlanStep(id="open", description="Open information in a new tab"),
                PlanStep(id="return", description="Return to the shop tab"),
            ],
        )

    async def decide(self, task, plan, observation, memory, tools, image=None) -> ModelDecision:
        if not self.opened:
            self.opened = True
            return ModelDecision(
                tool_call=ToolCall(
                    name="click_element",
                    arguments={"element_id": _element(observation, "Open information in new tab")},
                ),
                completed_step_ids=["open"],
            )
        if not self.read_other:
            if "Information tab" not in observation.text:
                raise AssertionError("The agent did not observe the new tab")
            self.read_other = True
            first_tab = next(tab.id for tab in observation.tabs if not tab.active)
            return ModelDecision(
                tool_call=ToolCall(name="switch_tab", arguments={"tab_id": first_tab}),
                completed_step_ids=["return"],
            )
        return ModelDecision(answer="I read the information tab and returned to the shop.")


class _UnexpectedProvider:
    label = "must-not-run"

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        raise AssertionError("A human verification page must not be planned through")

    async def decide(self, *args: object, **kwargs: object) -> ModelDecision:
        raise AssertionError("A human verification page must not reach the model")


class BrowserFlowTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        handler = partial(_QuietHandler, directory=str(FIXTURES))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.site_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    async def asyncSetUp(self) -> None:
        self.browser = BrowserController(allow_private_network=True)
        await self.browser.start(
            profile="temporary", headless=True, executable_path=BROWSER_EXECUTABLE
        )
        self.registry = ToolRegistry(self.browser)

    async def asyncTearDown(self) -> None:
        await self.browser.close()

    async def _execute(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
        *,
        approve=None,
    ) -> ToolResult:
        return await self.registry.execute(
            ToolCall(name=name, arguments=arguments or {}),
            mode=PermissionMode.SAFE,
            approve=approve,
        )

    async def _navigate(self, path: str) -> Observation:
        result = await self._execute("navigate", {"url": f"{self.site_url}/{path}"})
        self.assertTrue(result.success, result.message)
        return result.after_state or await self.browser.observe()

    async def test_human_verification_stops_without_solving(self) -> None:
        agent = AgentOrchestrator(
            _UnexpectedProvider(),
            browser=self.browser,
            registry=self.registry,
            permission_mode=PermissionMode.SAFE,
            max_steps=3,
        )
        result = await agent.run("Read the protected page", f"{self.site_url}/captcha.html")
        self.assertEqual(result.status, "blocked", result.error)
        self.assertIn("Human verification", result.answer)

    async def test_search_result_and_semantic_navigation(self) -> None:
        page = await self._navigate("index.html")
        result = await self._execute(
            "fill_element",
            {"element_id": _element(page, "Product search"), "text": "headphones"},
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.changed)

        page = result.after_state or await self.browser.observe()
        result = await self._execute(
            "click_element", {"element_id": _element(page, "Search products")}
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.changed)
        page = result.after_state or await self.browser.observe()
        self.assertIn("One result for headphones", page.text)
        self.assertIn("$79.95", page.text)

        result = await self._execute(
            "click_element", {"element_id": _element(page, "Open Studio headphones")}
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.changed)
        page = result.after_state or await self.browser.observe()
        self.assertTrue(page.url.endswith("/product.html"), page.url)
        self.assertIn("Price: $79.95", page.text)

    async def test_stale_element_rejected_then_reobserved_target_works(self) -> None:
        page = await self._navigate("index.html")
        stale_id = _element(page, "Continue")
        result = await self._execute(
            "click_element", {"element_id": _element(page, "Refresh options")}
        )
        self.assertTrue(result.success, result.message)

        stale = await self._execute("click_element", {"element_id": stale_id})
        self.assertFalse(stale.success)
        self.assertIn("stale", f"{stale.error_code} {stale.message}".lower())
        page = stale.after_state or await self.browser.observe()
        self.assertIn("Options refreshed", page.text)

        recovered = await self._execute("click_element", {"element_id": _element(page, "Continue")})
        self.assertTrue(recovered.success, recovered.message)
        page = recovered.after_state or await self.browser.observe()
        self.assertIn("New choice selected", page.text)

    async def test_no_op_click_has_no_verified_state_change(self) -> None:
        page = await self._navigate("index.html")
        result = await self._execute(
            "click_element", {"element_id": _element(page, "No-op control")}
        )
        self.assertFalse(result.changed, result.evidence)
        self.assertFalse(result.success, result.evidence)

    async def test_read_scroll_type_and_keyboard_search(self) -> None:
        page = await self._navigate("index.html")
        read = await self._execute("read_page")
        self.assertTrue(read.success, read.message)
        self.assertFalse(read.changed)
        self.assertIn("Fisher test shop", read.data["observation"]["text"])

        page = read.after_state or await self.browser.observe()
        typed = await self._execute(
            "type_text",
            {"element_id": _element(page, "Product search"), "text": "headphones"},
        )
        self.assertTrue(typed.success, typed.message)
        self.assertTrue(typed.changed)
        pressed = await self._execute("press_key", {"key": "Enter"}, approve=lambda _: True)
        self.assertTrue(pressed.success, pressed.message)
        self.assertIn("One result for headphones", pressed.after_state.text)

        scrolled = await self._execute("scroll", {"delta_y": 600})
        self.assertTrue(scrolled.success, scrolled.message)
        self.assertTrue(scrolled.changed)

    async def test_coordinate_click_fallback_requires_approval_and_works(self) -> None:
        page = await self._navigate("index.html")
        filled = await self._execute(
            "fill_element",
            {"element_id": _element(page, "Product search"), "text": "headphones"},
        )
        self.assertTrue(filled.success, filled.message)
        bounds = await self.browser.page.locator("#search-button").bounding_box()
        self.assertIsNotNone(bounds)
        point = {
            "x": int(bounds["x"] + bounds["width"] / 2),
            "y": int(bounds["y"] + bounds["height"] / 2),
        }
        denied = await self._execute("click_coordinates", point)
        self.assertFalse(denied.success)
        self.assertEqual(denied.error_code, "permission_required")
        clicked = await self._execute("click_coordinates", point, approve=lambda _: True)
        self.assertTrue(clicked.success, clicked.message)
        self.assertIn("One result for headphones", clicked.after_state.text)

    async def test_form_fill_respects_consequential_submit_permission(self) -> None:
        page = await self._navigate("form.html")
        for name, value in (
            ("Full name", "Taylor Fisher"),
            ("Email address", "taylor@example.test"),
            ("Delivery note", "Leave at desk"),
        ):
            result = await self._execute(
                "fill_element", {"element_id": _element(page, name), "text": value}
            )
            self.assertTrue(result.success, result.message)
            page = result.after_state or await self.browser.observe()
            self.assertEqual(next(item.value for item in page.elements if item.name == name), value)

        requests: list[PermissionRequest] = []

        def deny(request: PermissionRequest) -> bool:
            requests.append(request)
            return False

        denied = await self._execute(
            "click_element", {"element_id": _element(page, "Place order")}, approve=deny
        )
        self.assertFalse(denied.success)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].risk, RiskLevel.HIGH)
        page = denied.after_state or await self.browser.observe()
        self.assertIn("No order submitted", page.text)

        approved = await self._execute(
            "click_element",
            {"element_id": _element(page, "Place order")},
            approve=lambda request: True,
        )
        self.assertTrue(approved.success, approved.message)
        page = approved.after_state or await self.browser.observe()
        self.assertIn("Order submitted for Taylor Fisher", page.text)

    async def test_payment_field_requires_approval_and_is_redacted(self) -> None:
        page = await self._navigate("form.html")
        field = next(item for item in page.elements if item.name == "Card number")
        self.assertTrue(field.sensitive)
        request_call = {"element_id": field.id, "text": "4111111111111111"}
        denied = await self._execute("fill_element", request_call)
        self.assertEqual(denied.error_code, "permission_required")
        self.assertEqual(await self.browser.page.locator("#card-number").input_value(), "")
        requests: list[PermissionRequest] = []

        def approve(request: PermissionRequest) -> bool:
            requests.append(request)
            return True

        approved = await self._execute("fill_element", request_call, approve=approve)
        self.assertTrue(approved.success, approved.message)
        self.assertEqual(requests[0].risk, RiskLevel.HIGH)
        self.assertEqual(requests[0].call.arguments["text"], "[redacted]")
        self.assertEqual(
            await self.browser.page.locator("#card-number").input_value(), "4111111111111111"
        )
        self.assertEqual(
            next(
                item.value for item in approved.after_state.elements if item.name == "Card number"
            ),
            "[redacted]",
        )

    async def test_new_tab_switch_and_close(self) -> None:
        page = await self._navigate("index.html")
        first_tab = page.active_tab_id
        result = await self._execute(
            "click_element", {"element_id": _element(page, "Open information in new tab")}
        )
        self.assertTrue(result.success, result.message)
        page = result.after_state or await self.browser.observe()
        self.assertEqual(len(page.tabs), 2)
        second_tab = next(tab.id for tab in page.tabs if tab.id != first_tab)
        self.assertEqual(page.active_tab_id, second_tab)
        self.assertIn("Information tab", page.text)

        result = await self._execute("switch_tab", {"tab_id": first_tab})
        self.assertTrue(result.success, result.message)
        page = result.after_state or await self.browser.observe()
        self.assertEqual(page.active_tab_id, first_tab)
        self.assertIn("Fisher test shop", page.text)

        result = await self._execute("switch_tab", {"tab_id": second_tab})
        self.assertTrue(result.success, result.message)
        page = result.after_state or await self.browser.observe()
        self.assertEqual(page.active_tab_id, second_tab)
        self.assertIn("Information tab", page.text)

        result = await self._execute("close_tab", {"tab_id": second_tab})
        self.assertTrue(result.success, result.message)
        page = result.after_state or await self.browser.observe()
        self.assertEqual([tab.id for tab in page.tabs], [first_tab])

    async def test_agent_searches_opens_product_and_reports_observed_price(self) -> None:
        events: list[AgentEvent] = []
        agent = AgentOrchestrator(
            _ShopProvider(),
            browser=self.browser,
            registry=self.registry,
            permission_mode=PermissionMode.SAFE,
            on_event=events.append,
            max_steps=8,
        )
        result = await agent.run(
            "Find Studio headphones in the local shop and report their price",
            f"{self.site_url}/index.html",
        )
        self.assertEqual(result.status, "completed", result.error)
        self.assertIn("$79.95", result.answer)
        self.assertTrue(any(event.type == "tool_result" for event in events))
        self.assertTrue(any(event.type == "plan_updated" for event in events))
        self.assertTrue(self.browser.page.url.endswith("/product.html"))

    async def test_agent_recovers_after_stale_target(self) -> None:
        events: list[AgentEvent] = []
        provider = _StaleRecoveryProvider()
        agent = AgentOrchestrator(
            provider,
            browser=self.browser,
            registry=self.registry,
            permission_mode=PermissionMode.SAFE,
            on_event=events.append,
            max_steps=6,
        )
        result = await agent.run(
            "Refresh options and select the new choice", f"{self.site_url}/index.html"
        )
        self.assertEqual(result.status, "completed", result.error)
        self.assertTrue(provider.attempted_stale)
        self.assertTrue(
            any(
                event.type == "recovery" and event.data.get("strategy") == "reobserve"
                for event in events
            )
        )
        self.assertIn("New choice selected", (await self.browser.observe()).text)

    async def test_agent_stops_at_order_submission_permission_boundary(self) -> None:
        events: list[AgentEvent] = []
        agent = AgentOrchestrator(
            _FormProvider(),
            browser=self.browser,
            registry=self.registry,
            permission_mode=PermissionMode.SAFE,
            on_event=events.append,
            max_steps=7,
        )
        result = await agent.run("Fill the order form and submit it", f"{self.site_url}/form.html")
        self.assertEqual(result.status, "blocked", result.error)
        self.assertTrue(
            any(
                event.type == "tool_result"
                and event.data.get("error_code") == "permission_required"
                for event in events
            )
        )
        self.assertIn("No order submitted", (await self.browser.observe()).text)

    async def test_agent_inspects_new_tab_and_returns_to_first(self) -> None:
        provider = _TabProvider()
        agent = AgentOrchestrator(
            provider,
            browser=self.browser,
            registry=self.registry,
            permission_mode=PermissionMode.SAFE,
            max_steps=5,
        )
        result = await agent.run(
            "Open the information tab, inspect it, and return to the shop",
            f"{self.site_url}/index.html",
        )
        self.assertEqual(result.status, "completed", result.error)
        self.assertTrue(provider.read_other)
        page = await self.browser.observe()
        self.assertEqual(page.title, "Fisher test shop")
        self.assertEqual(len(page.tabs), 2)
