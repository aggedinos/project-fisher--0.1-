"""Conversation memory and state summarization (Improvement 2).

The agent keeps only the last few screenshots in the live history (older
turns lose their image). Everything older is folded into a single rolling
text summary produced by an extra LLM call, then surfaced to the model as a
"context from previous steps" block at the top of each prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm import LLMProvider, Message
from utils import parse_json_object

_SUMMARIZER_SYSTEM = (
    "You compress a browser agent's history into a compact running memory. "
    "Given the previous summary and new steps, return ONLY a JSON object: "
    '{"summary": "<concise paragraph>"}. The summary must preserve, for each '
    "step: the URL visited, the action taken, what was found, and the outcome. "
    "Keep concrete facts (names, numbers, quotes, URLs) the agent will need "
    "later. Be terse; drop chrome and navigation noise."
)


@dataclass
class StepRecord:
    """One executed step, used for summarization and the session log."""

    step: int
    url: str
    action: str
    params: dict[str, Any]
    reason: str
    confidence: int
    outcome: str

    def bullet(self) -> str:
        return (
            f"- Step {self.step} @ {self.url} | {self.action} {self.params} "
            f"(conf {self.confidence}) → {self.outcome}"
        )


class MemoryManager:
    """Caps screenshots and compresses old steps into a text summary."""

    def __init__(
        self,
        provider: LLMProvider,
        keep_recent: int,
        enabled: bool = True,
    ) -> None:
        self._provider = provider
        self._keep_recent = max(1, keep_recent)
        self._enabled = enabled
        self.records: list[StepRecord] = []
        self.summary: str = ""
        self._summarized_upto: int = 0  

    

    def record(
        self,
        *,
        step: int,
        url: str,
        action: str,
        params: dict[str, Any],
        reason: str,
        confidence: int,
        outcome: str,
    ) -> None:
        self.records.append(
            StepRecord(step, url, action, dict(params), reason, confidence, outcome)
        )

    

    async def compress(self) -> None:
        """Fold steps older than the recent window into ``self.summary``.

        Uses one short LLM call per compression. On any failure it falls back
        to plain-text bullets so memory still works offline.
        """
        cutoff = len(self.records) - self._keep_recent
        if cutoff <= self._summarized_upto:
            return
        new = self.records[self._summarized_upto : cutoff]
        bullets = "\n".join(r.bullet() for r in new)

        if not self._enabled:
            self.summary = (self.summary + "\n" + bullets).strip()
            self._summarized_upto = cutoff
            return

        prompt = (
            f"Previous summary:\n{self.summary or '(none)'}\n\n"
            f"New steps to fold in:\n{bullets}\n\n"
            'Return ONLY {"summary": "..."}.'
        )
        try:
            raw = await self._provider.generate(
                _SUMMARIZER_SYSTEM, [Message(role="user", text=prompt)]
            )
            merged = str(parse_json_object(raw).get("summary", "")).strip()
            self.summary = merged or (self.summary + "\n" + bullets).strip()
        except Exception:
            
            self.summary = (self.summary + "\n" + bullets).strip()
        self._summarized_upto = cutoff

    

    def context_text(self) -> str:
        """Block injected at the top of each prompt; '' when there's nothing."""
        if not self.records:
            return ""
        recent = self.records[self._summarized_upto :]
        parts: list[str] = ["CONTEXT FROM PREVIOUS STEPS"]
        if self.summary:
            parts.append(f"Summary so far:\n{self.summary}")
        if recent:
            parts.append(
                "Recent steps:\n" + "\n".join(r.bullet() for r in recent)
            )
        return "\n\n".join(parts)

    def keep_recent(self) -> int:
        return self._keep_recent
