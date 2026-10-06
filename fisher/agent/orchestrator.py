"""The observe, plan, act, verify browser-agent loop."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fisher.models import (
    AgentEvent,
    ModelDecision,
    Observation,
    PermissionMode,
    RunResult,
    TaskPlan,
    ToolCall,
    ToolResult,
    ToolSpec,
)

from .memory import MemoryManager, safe_call, safe_url
from .planner import Planner
from .recovery import (
    RecoveryDecision,
    RecoveryEngine,
    RecoveryStrategy,
    human_verification_required,
)
from .state import AgentState, Phase
from .verifier import verify_change


logger = logging.getLogger(__name__)
EventCallback = Callable[[AgentEvent], Any]
_ELEMENT_ID = re.compile(r"o\d+-e\d+\Z")
_TAB_ID = re.compile(r"t\d+\Z")
_PUBLIC_KEYS = {
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "Delete",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
}


def _public_call(call: ToolCall) -> dict[str, Any]:
    """Expose only safe, useful arguments to the local UI and event stream."""
    arguments: dict[str, Any] = {}
    for key, value in call.arguments.items():
        if key == "url" and isinstance(value, str):
            arguments[key] = safe_url(value)
        elif key == "element_id" and isinstance(value, str) and _ELEMENT_ID.fullmatch(value):
            arguments[key] = value
        elif key == "tab_id" and isinstance(value, str) and _TAB_ID.fullmatch(value):
            arguments[key] = value
        elif key == "key" and isinstance(value, str) and value in _PUBLIC_KEYS:
            arguments[key] = value
        elif key in {"delta_y", "x", "y"} and type(value) is int:
            arguments[key] = value
        else:
            arguments[key] = "[redacted]"
    return {"name": call.name, "arguments": arguments}


def _public_evidence(evidence: list[str]) -> list[str]:
    return [
        "URL changed" if item.startswith("URL changed:") else item[:180] for item in evidence[:5]
    ]


def _public_result(result: ToolResult) -> dict[str, Any]:
    return {
        "call": _public_call(result.call),
        "success": result.success,
        "changed": result.changed,
        "message": (
            "Action verified"
            if result.success and result.changed
            else "Page observed"
            if result.success
            else (result.error_code or "Action failed").replace("_", " ").capitalize()
        ),
        "evidence": _public_evidence(result.evidence),
        "error_code": result.error_code,
        "retryable": result.retryable,
    }


class AgentOrchestrator:
    """Run one task with explicit observations and bounded recovery.

    A passed browser belongs to its caller and must already be started. The
    orchestrator creates and closes a browser only when none was passed.
    Every navigation, including the initial URL, goes through ToolRegistry.
    """

    def __init__(
        self,
        provider: Any,
        browser: Any | None = None,
        registry: Any | None = None,
        permission_mode: PermissionMode = PermissionMode.SAFE,
        approve: Callable[..., Any] | None = None,
        on_event: EventCallback | None = None,
        recorder: Any | None = None,
        max_steps: int = 30,
        max_replans: int = 2,
        max_decision_failures: int = 3,
        record_screenshots: bool = False,
    ) -> None:
        if registry is not None and browser is None:
            raise ValueError("A registry requires its browser to be passed too")
        self.provider = provider
        self.browser = browser
        self.registry = registry
        self.permission_mode = PermissionMode(permission_mode)
        self.approve = approve
        self.on_event = on_event
        self.recorder = recorder
        self.max_steps = max(1, max_steps)
        self.max_replans = max(0, max_replans)
        self.max_decision_failures = max(1, max_decision_failures)
        self.record_screenshots = record_screenshots
        self._cancelled = asyncio.Event()
        self._run_task: asyncio.Task[Any] | None = None
        self._state: AgentState | None = None

    @property
    def state(self) -> AgentState | None:
        return self._state

    def cancel(self) -> None:
        """Request stop and interrupt an outstanding model or browser await."""
        self._cancelled.set()
        task = self._run_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _emit(self, kind: str, **data: Any) -> None:
        state = self._state
        if state is None:
            return
        event = AgentEvent(type=kind, task_id=state.task_id, data=data)
        if self.recorder is not None:
            try:
                self.recorder.record_event(event)
            except Exception as exc:
                logger.error("Could not record agent event %s (%s)", kind, type(exc).__name__)
        if self.on_event is not None:
            try:
                outcome = self.on_event(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as exc:
                logger.error("Agent event callback failed for %s (%s)", kind, type(exc).__name__)

    async def _phase(self, phase: Phase) -> None:
        assert self._state is not None
        self._state.phase = phase
        await self._emit("state_changed", phase=phase.value, step=self._state.steps)
        messages = {
            Phase.PLAN: "Planning the task",
            Phase.DECIDE: "Choosing the next browser action",
            Phase.ACT: "Using the browser",
            Phase.VERIFY: "Checking the result",
            Phase.RECOVER: "Trying another approach",
        }
        if phase in messages:
            await self._emit("status", message=messages[phase])

    async def _take_screenshot(self) -> bytes | None:
        try:
            return await self.browser.screenshot()
        except Exception as exc:
            logger.warning("Could not capture agent screenshot (%s)", type(exc).__name__)
            return None

    async def _observe(self) -> Observation:
        await self._phase(Phase.OBSERVE)
        observation: Observation = await self.browser.observe()
        assert self._state is not None
        self._state.observe(observation)
        await self._emit(
            "observation_updated",
            observation={"url": safe_url(observation.url), "title": observation.title[:180]},
            url=safe_url(observation.url),
            title=observation.title[:180],
            element_count=len(observation.elements),
            tab_count=len(observation.tabs),
        )
        return observation

    async def _execute(self, call: ToolCall, before: Observation) -> ToolResult:
        """Validate and execute through the registry, then check observed state."""
        assert self.registry is not None
        await self._phase(Phase.VALIDATE)
        await self._emit("tool_selected", tool=call.name, detail=safe_call(call))
        await self._emit("tool_started", call=_public_call(call))
        if self._cancelled.is_set():
            raise asyncio.CancelledError
        await self._phase(Phase.ACT)
        try:
            result: ToolResult = await self.registry.execute(
                call, mode=self.permission_mode, approve=self.approve
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Tool execution failed: %s (%s)", call.name, type(exc).__name__)
            result = ToolResult(
                call=call,
                success=False,
                message=f"Tool raised {type(exc).__name__}",
                error_code="tool_exception",
                retryable=False,
                before_state=before,
            )

        await self._phase(Phase.VERIFY)
        if result.before_state is None:
            result.before_state = before
        if result.after_state is None:
            try:
                result.after_state = await self.browser.observe()
            except Exception as exc:
                logger.warning(
                    "Could not observe browser after %s (%s)", call.name, type(exc).__name__
                )
        if result.success and result.before_state is not None and result.after_state is not None:
            verified, evidence = verify_change(call, result.before_state, result.after_state)
            result.evidence = list(dict.fromkeys(result.evidence + evidence))
            if not verified:
                result.success = False
                result.changed = False
                result.error_code = "verification_failed"
                result.message = evidence[-1] if evidence else "Action did not change page"
            elif call.name not in {"read_page", "screenshot"}:
                result.changed = True
        elif result.success:
            result.success = False
            result.error_code = "verification_unavailable"
            result.message = "Could not observe state after action"
        await self._emit(
            "tool_result",
            tool=call.name,
            success=result.success,
            changed=result.changed,
            error_code=result.error_code,
            evidence=_public_evidence(result.evidence),
        )
        await self._emit("tool_finished", result=_public_result(result))
        await self._emit(
            "verification_result",
            changed=result.changed if result.success else False,
            evidence=_public_evidence(result.evidence),
        )
        if result.after_state is not None:
            assert self._state is not None
            self._state.observe(result.after_state)
        return result

    def _specs(self) -> list[ToolSpec]:
        assert self.registry is not None
        return list(self.registry.specs())

    def _retry_allowed(self, name: str) -> bool:
        # A timed-out click, key press, or form entry may already have taken
        # effect. Repeating it automatically could submit or type twice.
        if name not in {"navigate", "switch_tab", "read_page"}:
            return False
        return any(spec.name == name and spec.retryable for spec in self._specs())

    async def _record_step(
        self,
        call: ToolCall,
        result: ToolResult,
        plan: TaskPlan,
        screenshot: bytes | None,
    ) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.record_step(
                call,
                result,
                plan=plan,
                screenshot=screenshot if self.record_screenshots else None,
            )
        except Exception as exc:
            logger.error("Could not record agent step (%s)", type(exc).__name__)

    async def _replan(
        self,
        planner: Planner,
        state: AgentState,
        reason: str,
    ) -> bool:
        if state.replans >= self.max_replans:
            await self._emit("recovery", strategy="stop", reason="replan budget exhausted")
            return False
        state.replans += 1
        await self._phase(Phase.RECOVER)
        await self._emit("recovery", strategy="replan", reason=reason)
        observation = await self._observe()
        assert state.plan is not None
        state.plan = await planner.replan(state.task, observation, state.plan)
        await self._emit("plan_updated", plan=state.plan.model_dump())
        return True

    async def run(self, task: str, start_url: str, task_id: str | None = None) -> RunResult:
        """Execute a task, returning a completed, partial, blocked or stopped result."""
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("This agent is already running")
        self._run_task = asyncio.current_task()
        state = AgentState(task_id=task_id or uuid4().hex, task=task, start_url=start_url)
        self._state = state
        owns_browser = self.browser is None
        memory = MemoryManager()
        planner = Planner(self.provider)
        recovery = RecoveryEngine()
        final = RunResult(task_id=state.task_id, status="failed", error="Task did not start")
        try:
            if self.recorder is not None:
                self.recorder.start(
                    task, str(getattr(self.provider, "label", type(self.provider).__name__))
                )
            await self._emit("started", task=task, start_url=safe_url(start_url))
            await self._emit("agent_started", task=task, start_url=safe_url(start_url))
            if self._cancelled.is_set():
                raise asyncio.CancelledError

            if owns_browser:
                from fisher.browser.controller import BrowserController

                self.browser = BrowserController()
                await self.browser.start()
            if self.registry is None:
                from fisher.tools.registry import ToolRegistry

                self.registry = ToolRegistry(self.browser)

            initial = await self._observe()
            navigation = ToolCall(name="navigate", arguments={"url": start_url})
            initial_result = await self._execute(navigation, initial)
            if not initial_result.success:
                await self._emit(
                    "failed",
                    reason="initial_navigation",
                    message=(initial_result.error_code or "navigation failed"),
                    error_code=initial_result.error_code,
                )
                final = RunResult(
                    task_id=state.task_id,
                    status="blocked"
                    if initial_result.error_code
                    in {
                        "permission_denied",
                        "permission_required",
                        "policy_denied",
                        "unsafe_url",
                        "blocked_url",
                        "approval_denied",
                    }
                    else "failed",
                    error=f"Initial navigation failed: {initial_result.error_code or 'browser error'}",
                )
                return final

            observation = initial_result.after_state or await self._observe()
            state.observe(observation)
            memory.observe(observation)
            if human_verification_required(observation):
                state.answer = "Human verification is required. Complete it manually or use another legitimate source."
                await self._emit("status", message="Human verification required")
                final = RunResult(
                    task_id=state.task_id,
                    status="blocked",
                    answer=state.answer,
                    error="Human verification required",
                )
                return final
            await self._phase(Phase.PLAN)
            state.plan = await planner.create(task, observation)
            await self._emit("plan_updated", plan=state.plan.model_dump())

            pending_retry: ToolCall | None = None
            decision_failures = 0
            while state.steps < self.max_steps:
                if self._cancelled.is_set():
                    raise asyncio.CancelledError
                observation = await self._observe()
                if human_verification_required(observation):
                    state.answer = "Human verification is required. Complete it manually or use another legitimate source."
                    await self._emit("status", message="Human verification required")
                    final = RunResult(
                        task_id=state.task_id,
                        status="blocked",
                        answer=state.answer,
                        steps=state.steps,
                        error="Human verification required",
                    )
                    break
                memory.observe(observation)
                assert state.plan is not None

                decision: ModelDecision | None = None
                if pending_retry is None:
                    await self._phase(Phase.DECIDE)
                    image = (
                        await self._take_screenshot()
                        if getattr(self.provider, "send_images", True)
                        else None
                    )
                    try:
                        decision = await self.provider.decide(
                            task,
                            state.plan,
                            observation,
                            memory.context_text(),
                            self._specs(),
                            image=image,
                        )
                        if not isinstance(decision, ModelDecision):
                            decision = ModelDecision.model_validate(decision)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("Model decision failed (%s)", type(exc).__name__)
                        decision_failures += 1
                        await self._emit(
                            "recovery",
                            strategy="reobserve",
                            reason="model_decision_failed",
                            attempt=decision_failures,
                        )
                        if decision_failures >= self.max_decision_failures:
                            final = RunResult(
                                task_id=state.task_id,
                                status="failed",
                                steps=state.steps,
                                error=f"Model failed to return a valid action ({type(exc).__name__})",
                            )
                            break
                        if decision_failures == 2:
                            await self._replan(planner, state, "invalid model output")
                        continue
                if self._cancelled.is_set():
                    raise asyncio.CancelledError

                if decision is not None:
                    memory.add_facts(decision.facts, observation.url)
                    if decision.answer and not decision.tool_call:
                        answer = decision.answer.strip()
                        if answer and (
                            observation.text.strip()
                            or observation.elements
                            or state.verified_actions > 0
                        ):
                            state.answer = answer
                            Planner.mark_completed(state.plan, decision.completed_step_ids)
                            await self._phase(Phase.COMPLETE)
                            await self._emit(
                                "completed",
                                answer=answer,
                                evidence={
                                    "source_url": safe_url(observation.url),
                                    "verified_actions": state.verified_actions,
                                    "observed_text": bool(observation.text.strip()),
                                },
                            )
                            final = RunResult(
                                task_id=state.task_id,
                                status="completed",
                                answer=answer,
                                steps=state.steps,
                            )
                            break
                        await self._emit(
                            "recovery",
                            strategy="reobserve",
                            reason="answer lacks observed evidence",
                        )
                        decision_failures += 1
                        if decision_failures >= self.max_decision_failures:
                            final = RunResult(
                                task_id=state.task_id,
                                status="failed",
                                steps=state.steps,
                                error="Answer lacked observed evidence",
                            )
                            break
                        continue
                    call = decision.tool_call
                    if call is None:
                        decision_failures += 1
                        await self._emit(
                            "recovery",
                            strategy="reobserve",
                            reason="model supplied neither tool nor answer",
                        )
                        if decision_failures >= self.max_decision_failures:
                            final = RunResult(
                                task_id=state.task_id,
                                status="failed",
                                steps=state.steps,
                                error="Model supplied neither tool nor answer",
                            )
                            break
                        continue
                    decision_failures = 0
                else:
                    call = pending_retry
                    pending_retry = None
                assert call is not None

                loop_guard = recovery.before_action(call, observation)
                if loop_guard.strategy == RecoveryStrategy.REPLAN:
                    if not await self._replan(planner, state, loop_guard.reason):
                        final = RunResult(
                            task_id=state.task_id,
                            status="partial",
                            steps=state.steps,
                            error="Repeated browser actions without progress",
                        )
                        break
                    continue

                state.steps += 1
                screenshot = await self._take_screenshot() if self.record_screenshots else None
                result = await self._execute(call, observation)
                state.last_result = result
                memory.record(state.steps, call, result)
                await self._record_step(call, result, state.plan, screenshot)
                if result.success:
                    state.verified_actions += 1
                    if decision is not None:
                        completed = Planner.mark_completed(state.plan, decision.completed_step_ids)
                        if completed:
                            await self._emit(
                                "plan_updated",
                                plan=state.plan.model_dump(),
                                completed_step_ids=completed,
                            )

                recovery_choice: RecoveryDecision = recovery.after_action(
                    call,
                    observation,
                    result,
                    retry_allowed=self._retry_allowed(call.name),
                )
                if recovery_choice.strategy == RecoveryStrategy.CONTINUE:
                    continue
                await self._phase(Phase.RECOVER)
                await self._emit(
                    "recovery",
                    strategy=recovery_choice.strategy.value,
                    reason=recovery_choice.reason,
                    tool=call.name,
                )
                if recovery_choice.strategy == RecoveryStrategy.RETRY:
                    pending_retry = call
                elif recovery_choice.strategy == RecoveryStrategy.REPLAN:
                    recovery.consecutive_failures = 0
                    if not await self._replan(planner, state, recovery_choice.reason):
                        final = RunResult(
                            task_id=state.task_id,
                            status="partial",
                            steps=state.steps,
                            error="Recovery budget exhausted",
                        )
                        break
                elif recovery_choice.strategy == RecoveryStrategy.STOP:
                    if result.error_code in {"permission_required", "permission_denied"}:
                        state.answer = (
                            "The requested action was not approved; no action was executed."
                        )
                    final = RunResult(
                        task_id=state.task_id,
                        status="blocked",
                        answer=state.answer,
                        steps=state.steps,
                        error=recovery_choice.reason,
                    )
                    break
            else:
                final = RunResult(
                    task_id=state.task_id,
                    status="partial",
                    steps=state.steps,
                    error=f"Reached the {self.max_steps}-action limit",
                )
        except asyncio.CancelledError:
            final = RunResult(
                task_id=state.task_id,
                status="stopped",
                answer=state.answer,
                steps=state.steps,
                error="Cancelled by user",
            )
            await self._emit("stopped", reason="cancelled")
        except Exception as exc:
            logger.error(
                "Agent run %s failed in %s (%s)",
                state.task_id,
                state.phase.value,
                type(exc).__name__,
            )
            final = RunResult(
                task_id=state.task_id,
                status="failed",
                answer=state.answer,
                steps=state.steps,
                error=f"Agent failed ({type(exc).__name__})",
            )
            await self._emit("agent_error", message=f"Agent failed ({type(exc).__name__})")
        finally:
            await self._emit("agent_finished", result=final.model_dump())
            if self.recorder is not None:
                try:
                    self.recorder.finish(final.status, final.answer)
                except Exception as exc:
                    logger.error("Could not finish session recording (%s)", type(exc).__name__)
            if owns_browser and self.browser is not None:
                try:
                    await self.browser.close()
                except Exception as exc:
                    logger.error("Could not close owned browser (%s)", type(exc).__name__)
            self._run_task = None
        return final
