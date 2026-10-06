"""Structured planning and progress tracking."""

from __future__ import annotations

import logging
from typing import Protocol

from fisher.models import Observation, PlanStep, TaskPlan


logger = logging.getLogger(__name__)


class PlanningProvider(Protocol):
    async def plan(self, task: str, observation: Observation) -> TaskPlan: ...


def _normalize(plan: TaskPlan, goal: str, revision: int) -> TaskPlan:
    """Keep provider plans small, stable and free from duplicate IDs."""
    seen: set[str] = set()
    steps: list[PlanStep] = []
    for index, item in enumerate(plan.steps[:12], start=1):
        description = item.description.strip()[:300]
        if not description:
            continue
        identifier = item.id.strip()[:64] or f"step-{index}"
        if identifier in seen:
            identifier = f"{identifier}-{index}"
        seen.add(identifier)
        steps.append(PlanStep(id=identifier, description=description, done=bool(item.done)))
    if not steps:
        steps = [PlanStep(id="complete-task", description="Complete the user's task")]
    normalized = TaskPlan(
        goal=goal,
        steps=steps,
        assumptions=[value[:300] for value in plan.assumptions[:8] if value.strip()],
        revision=revision,
    )
    normalized.current_step = next(
        (index for index, step in enumerate(steps) if not step.done), len(steps)
    )
    return normalized


class Planner:
    def __init__(self, provider: PlanningProvider) -> None:
        self.provider = provider

    async def create(self, task: str, observation: Observation) -> TaskPlan:
        try:
            plan = await self.provider.plan(task, observation)
            return _normalize(plan, task, 1)
        except Exception as exc:
            logger.warning(
                "Could not create model plan (%s); using a short fallback", type(exc).__name__
            )
            return _normalize(TaskPlan(goal=task), task, 1)

    async def replan(self, task: str, observation: Observation, previous: TaskPlan) -> TaskPlan:
        """Regenerate pending objectives while retaining completed work."""
        try:
            fresh = _normalize(
                await self.provider.plan(task, observation), task, previous.revision + 1
            )
        except Exception as exc:
            logger.warning("Could not replan (%s); keeping the current plan", type(exc).__name__)
            return previous

        completed = [item.model_copy() for item in previous.steps if item.done]
        completed_descriptions = {item.description.casefold() for item in completed}
        pending = [
            item
            for item in fresh.steps
            if item.description.casefold() not in completed_descriptions
        ]
        used = {item.id for item in completed}
        for index, item in enumerate(pending, start=1):
            if item.id in used:
                item.id = f"replan-{fresh.revision}-{index}"
            used.add(item.id)
        fresh.steps = completed + pending
        fresh.current_step = next(
            (index for index, step in enumerate(fresh.steps) if not step.done),
            len(fresh.steps),
        )
        return fresh

    @staticmethod
    def mark_completed(plan: TaskPlan, step_ids: list[str]) -> list[str]:
        """Mark only model-named steps, after a verified browser action."""
        requested = set(step_ids)
        changed: list[str] = []
        for item in plan.steps:
            if item.id in requested and not item.done:
                item.done = True
                changed.append(item.id)
        plan.current_step = next(
            (index for index, step in enumerate(plan.steps) if not step.done),
            len(plan.steps),
        )
        return changed
