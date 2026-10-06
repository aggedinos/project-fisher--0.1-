"""State kept for one browser-agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from fisher.models import Observation, TaskPlan, ToolResult


class Phase(str, Enum):
    UNDERSTAND = "understand"
    PLAN = "plan"
    OBSERVE = "observe"
    DECIDE = "decide"
    VALIDATE = "validate"
    ACT = "act"
    VERIFY = "verify"
    RECOVER = "recover"
    COMPLETE = "complete"


@dataclass
class AgentState:
    task_id: str
    task: str
    start_url: str
    phase: Phase = Phase.UNDERSTAND
    plan: TaskPlan | None = None
    observation: Observation | None = None
    last_result: ToolResult | None = None
    steps: int = 0
    verified_actions: int = 0
    replans: int = 0
    answer: str = ""
    visited_urls: set[str] = field(default_factory=set)

    def observe(self, observation: Observation) -> None:
        self.observation = observation
        if observation.url:
            self.visited_urls.add(observation.url)
