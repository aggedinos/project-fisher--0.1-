"""Task planning and replanning (Improvement 7).

A separate LLM call turns the task + start URL into a short ordered plan that
is then injected into every agent prompt so the agent stays on track. If the
agent stalls, ``replan`` regenerates the plan from the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm import LLMProvider, Message
from utils import parse_json_object

_PLANNER_SYSTEM = (
    "You are a planning module for a web browsing agent. Given a task and a "
    "starting URL, produce a short, ordered, high-level plan. Return ONLY a "
    'JSON object: {"steps": ["...", "..."], "estimated_actions": <int>}. '
    "Each step is one short imperative sentence. Aim for 3-8 steps. "
    "estimated_actions is a rough integer guess of total browser actions."
)


@dataclass
class Plan:
    """An ordered plan the agent follows and can replan."""

    steps: list[str] = field(default_factory=list)
    estimated_actions: int = 0
    revision: int = 1

    def as_prompt_block(self) -> str:
        if not self.steps:
            return ""
        lines = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
        return (
            f"YOUR OVERALL PLAN (revision {self.revision}, "
            f"~{self.estimated_actions} actions):\n{lines}"
        )

    def pretty(self) -> str:
        head = (
            f"Plan (revision {self.revision}, "
            f"~{self.estimated_actions} actions):"
        )
        body = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
        return f"{head}\n{body}"


def _coerce(raw: str, revision: int) -> Plan:
    try:
        data = parse_json_object(raw)
        steps = [str(s).strip() for s in data.get("steps", []) if str(s).strip()]
        est = int(data.get("estimated_actions", len(steps) * 3) or len(steps) * 3)
    except Exception:
        steps, est = [], 0
    return Plan(steps=steps, estimated_actions=est, revision=revision)


async def make_plan(
    provider: LLMProvider, task: str, start_url: str
) -> Plan:
    """Create the initial plan before the agent loop starts."""
    prompt = (
        f"Task: {task}\nStart URL: {start_url}\n\n"
        'Return ONLY {"steps": [...], "estimated_actions": <int>}.'
    )
    try:
        raw = await provider.generate(
            _PLANNER_SYSTEM, [Message(role="user", text=prompt)]
        )
    except Exception:
        return Plan(steps=[], estimated_actions=0, revision=1)
    return _coerce(raw, revision=1)


async def replan(
    provider: LLMProvider,
    task: str,
    current_url: str,
    progress_summary: str,
    revision: int,
) -> Plan:
    """Regenerate the plan from the current state after a stall."""
    prompt = (
        f"Task: {task}\nCurrent URL: {current_url}\n\n"
        f"What has happened so far:\n{progress_summary or '(little progress)'}\n\n"
        "The agent is stuck. Produce a NEW plan from the CURRENT state to "
        'finish the task. Return ONLY {"steps": [...], "estimated_actions": <int>}.'
    )
    try:
        raw = await provider.generate(
            _PLANNER_SYSTEM, [Message(role="user", text=prompt)]
        )
    except Exception:
        return Plan(steps=[], estimated_actions=0, revision=revision)
    return _coerce(raw, revision=revision)
