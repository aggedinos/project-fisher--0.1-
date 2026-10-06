"""Small, serializable contracts shared across the application."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PermissionMode(str, Enum):
    AUTO = "auto"
    SAFE = "safe"
    SUPERVISED = "supervised"


class ElementInfo(BaseModel):
    id: str
    role: str
    name: str = ""
    tag: str = ""
    text: str = ""
    visible: bool = True
    enabled: bool = True
    href: str | None = None
    value: str | None = None
    input_type: str | None = None
    sensitive: bool = False


class TabInfo(BaseModel):
    id: str
    url: str
    title: str = ""
    active: bool = False


class Observation(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    elements: list[ElementInfo] = Field(default_factory=list)
    tabs: list[TabInfo] = Field(default_factory=list)
    active_tab_id: str = ""
    fingerprint: str = ""

    def summary(self, max_text: int = 4000) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text[:max_text],
            "elements": [item.model_dump(exclude_none=True) for item in self.elements],
            "tabs": [item.model_dump() for item in self.tabs],
            "active_tab_id": self.active_tab_id,
        }


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    risk: RiskLevel = RiskLevel.LOW
    timeout_seconds: float = 15.0
    retryable: bool = False


class ToolResult(BaseModel):
    call: ToolCall
    success: bool
    changed: bool = False
    message: str = ""
    evidence: list[str] = Field(default_factory=list)
    error_code: str | None = None
    retryable: bool = False
    before_state: Observation | None = None
    after_state: Observation | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ModelDecision(BaseModel):
    tool_call: ToolCall | None = None
    answer: str | None = None
    facts: list[str] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    id: str
    description: str
    done: bool = False


class TaskPlan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    assumptions: list[str] = Field(default_factory=list)
    revision: int = 0


class PermissionRequest(BaseModel):
    id: str
    call: ToolCall
    risk: RiskLevel
    reason: str


class AgentEvent(BaseModel):
    type: str
    task_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunResult(BaseModel):
    task_id: str
    status: str
    answer: str = ""
    steps: int = 0
    error: str | None = None
