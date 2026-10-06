"""Focused tests for the new agent's verification and control flow."""

from __future__ import annotations

import asyncio
import unittest

from fisher.agent.memory import MemoryManager
from fisher.agent.orchestrator import AgentOrchestrator
from fisher.agent.planner import Planner
from fisher.agent.verifier import verify_change
from fisher.models import (
    ElementInfo,
    ModelDecision,
    Observation,
    PlanStep,
    TaskPlan,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class VerificationTests(unittest.TestCase):
    def test_transient_element_ids_do_not_count_as_progress(self) -> None:
        before = Observation(
            url="https://example.com",
            text="Shop",
            fingerprint="same",
            elements=[ElementInfo(id="o1-e1", role="button", name="Search")],
        )
        after = before.model_copy(deep=True)
        after.elements[0].id = "o2-e1"
        verified, evidence = verify_change(
            ToolCall(name="click_element", arguments={"element_id": "o1-e1"}),
            before,
            after,
        )
        self.assertFalse(verified)
        self.assertIn("no observable page change", evidence[0])

    def test_filled_value_is_observable(self) -> None:
        before = Observation(
            url="https://example.com",
            fingerprint="before",
            elements=[ElementInfo(id="o1-e1", role="textbox", name="Search", value="")],
        )
        after = Observation(
            url="https://example.com",
            fingerprint="after",
            elements=[ElementInfo(id="o2-e1", role="textbox", name="Search", value="headphones")],
        )
        verified, evidence = verify_change(
            ToolCall(name="fill_element", arguments={"element_id": "o1-e1", "text": "headphones"}),
            before,
            after,
        )
        self.assertTrue(verified)
        self.assertIn("interactive element state changed", evidence)

    def test_plan_advances_only_explicit_step_ids(self) -> None:
        plan = TaskPlan(
            goal="Find a price",
            steps=[
                PlanStep(id="search", description="Search"),
                PlanStep(id="report", description="Report price"),
            ],
        )
        self.assertEqual(Planner.mark_completed(plan, ["not-a-step"]), [])
        self.assertEqual(plan.current_step, 0)
        self.assertEqual(Planner.mark_completed(plan, ["search"]), ["search"])
        self.assertEqual(plan.current_step, 1)

    def test_memory_never_repeats_typed_text(self) -> None:
        memory = MemoryManager()
        call = ToolCall(name="fill_element", arguments={"element_id": "e1", "text": "secret-value"})
        observation = Observation(url="https://example.com/form")
        result = ToolResult(call=call, success=True, changed=True, after_state=observation)
        memory.record(1, call, result)
        context = memory.context_text()
        self.assertNotIn("secret-value", context)
        self.assertIn("[redacted]", context)


class _Browser:
    def __init__(self) -> None:
        self.current = Observation(url="about:blank", fingerprint="blank")
        self.closed = False

    async def observe(self) -> Observation:
        return self.current.model_copy(deep=True)

    async def screenshot(self) -> bytes:
        return b"jpeg"

    async def close(self) -> None:
        self.closed = True


class _Registry:
    def __init__(self, browser: _Browser) -> None:
        self.browser = browser
        self.calls: list[ToolCall] = []

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name="navigate", description="Open URL", parameters={}, retryable=True),
            ToolSpec(name="read_page", description="Read page", parameters={}),
        ]

    async def execute(self, call: ToolCall, **_kwargs: object) -> ToolResult:
        self.calls.append(call)
        before = await self.browser.observe()
        if call.name == "navigate":
            self.browser.current = Observation(
                url=str(call.arguments["url"]),
                title="Product",
                text="Headphones cost $49",
                fingerprint="product",
            )
        after = await self.browser.observe()
        return ToolResult(
            call=call,
            success=True,
            changed=call.name == "navigate",
            before_state=before,
            after_state=after,
        )


class _Provider:
    label = "fake"

    def __init__(self) -> None:
        self.decisions = 0

    async def plan(self, task: str, _observation: Observation) -> TaskPlan:
        return TaskPlan(goal=task, steps=[PlanStep(id="read", description="Read product")])

    async def decide(
        self,
        _task: str,
        _plan: TaskPlan,
        _observation: Observation,
        _memory: str,
        _tools: list[ToolSpec],
        image: bytes | None = None,
    ) -> ModelDecision:
        self.decisions += 1
        if self.decisions == 1:
            return ModelDecision(tool_call=ToolCall(name="read_page"), completed_step_ids=["read"])
        return ModelDecision(answer="The headphones cost $49.", facts=["Price: $49"])


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_plan_read_and_complete(self) -> None:
        browser = _Browser()
        registry = _Registry(browser)
        events = []
        agent = AgentOrchestrator(
            _Provider(),
            browser=browser,
            registry=registry,
            on_event=events.append,
            max_steps=4,
        )
        result = await agent.run(
            "Report the price", "https://example.com/product", task_id="case-1"
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "The headphones cost $49.")
        self.assertEqual([call.name for call in registry.calls], ["navigate", "read_page"])
        self.assertTrue(agent.state.plan.steps[0].done)
        self.assertIn("observation_updated", [event.type for event in events])
        self.assertEqual(events[-1].type, "agent_finished")
        self.assertEqual(events[-1].data["result"]["status"], "completed")
        self.assertFalse(browser.closed, "Caller-owned browser must remain open")

    async def test_cancellation_interrupts_provider_and_finishes(self) -> None:
        entered = asyncio.Event()

        class WaitingProvider(_Provider):
            async def decide(self, *_args: object, **_kwargs: object) -> ModelDecision:
                entered.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        browser = _Browser()
        events = []
        agent = AgentOrchestrator(
            WaitingProvider(),
            browser=browser,
            registry=_Registry(browser),
            on_event=events.append,
        )
        task = asyncio.create_task(agent.run("Wait", "https://example.com/product"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        agent.cancel()
        result = await asyncio.wait_for(task, timeout=2)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(events[-1].type, "agent_finished")
        self.assertFalse(browser.closed)


if __name__ == "__main__":
    unittest.main()
