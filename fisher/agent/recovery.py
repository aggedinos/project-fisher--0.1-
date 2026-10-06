"""Bounded recovery choices for failed browser interactions."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import Enum

from fisher.models import Observation, ToolCall, ToolResult


def human_verification_required(observation: Observation) -> bool:
    """Recognize common challenge screens without attempting to solve them."""
    content = f"{observation.title} {observation.text[:3000]}".casefold()
    return any(
        phrase in content
        for phrase in (
            "verify you are human",
            "complete the captcha",
            "solve the captcha",
            "i'm not a robot",
            "i am not a robot",
            "human verification required",
            "checking your browser before accessing",
        )
    )


class RecoveryStrategy(str, Enum):
    CONTINUE = "continue"
    REOBSERVE = "reobserve"
    RETRY = "retry"
    REPLAN = "replan"
    STOP = "stop"


@dataclass(frozen=True)
class RecoveryDecision:
    strategy: RecoveryStrategy
    reason: str


class RecoveryEngine:
    """Track unchanged actions, loops and retry budgets for one run."""

    def __init__(
        self,
        max_same_action: int = 2,
        max_retry_per_action: int = 1,
        max_consecutive_failures: int = 4,
    ) -> None:
        self.max_same_action = max_same_action
        self.max_retry_per_action = max_retry_per_action
        self.max_consecutive_failures = max_consecutive_failures
        self.recent: deque[str] = deque(maxlen=8)
        self.retries: dict[str, int] = {}
        self.consecutive_failures = 0

    @staticmethod
    def signature(call: ToolCall, observation: Observation) -> str:
        args = json.dumps(call.arguments, sort_keys=True, default=str)
        return f"{observation.url}|{observation.fingerprint}|{call.name}|{args}"

    def before_action(self, call: ToolCall, observation: Observation) -> RecoveryDecision:
        signature = self.signature(call, observation)
        if len(self.recent) >= self.max_same_action and all(
            value == signature for value in list(self.recent)[-self.max_same_action:]
        ):
            return RecoveryDecision(RecoveryStrategy.REPLAN, "identical action on unchanged page")
        if len(self.recent) >= 4:
            tail = list(self.recent)[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and signature == tail[0]:
                return RecoveryDecision(RecoveryStrategy.REPLAN, "alternating action loop")
        return RecoveryDecision(RecoveryStrategy.CONTINUE, "")

    def after_action(
        self, call: ToolCall, before: Observation, result: ToolResult,
        *, retry_allowed: bool,
    ) -> RecoveryDecision:
        signature = self.signature(call, before)
        self.recent.append(signature)
        if result.success:
            self.consecutive_failures = 0
            self.retries.pop(signature, None)
            return RecoveryDecision(RecoveryStrategy.CONTINUE, "action verified")

        self.consecutive_failures += 1
        code = (result.error_code or "").lower()
        if code in {"permission_denied", "permission_required", "policy_denied",
                    "unsafe_url", "blocked_url"}:
            return RecoveryDecision(RecoveryStrategy.STOP, result.message or code)
        if self.consecutive_failures >= self.max_consecutive_failures:
            return RecoveryDecision(RecoveryStrategy.REPLAN, "several consecutive failures")
        if code in {"stale_element", "element_not_found", "element_stale"}:
            return RecoveryDecision(RecoveryStrategy.REOBSERVE, "element reference became stale")
        if retry_allowed and result.retryable:
            count = self.retries.get(signature, 0)
            if count < self.max_retry_per_action:
                self.retries[signature] = count + 1
                return RecoveryDecision(RecoveryStrategy.RETRY, "transient tool failure")
        return RecoveryDecision(RecoveryStrategy.REOBSERVE, result.message or "action did not change page")
