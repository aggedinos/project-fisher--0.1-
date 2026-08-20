"""Session recording and replay (Improvement 8).

Every step (screenshot, action, confidence, result, timestamp) is recorded.
At the end of a run the session is written to ``sessions/`` as both a JSON
file (machine-readable / replayable) and a human-readable ``*_report.txt``.
``replay()`` re-executes a saved session visually for debugging.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable

from utils import ensure_dir


@dataclass
class _Step:
    step: int
    timestamp: str
    url: str
    action: dict[str, Any]
    confidence: int
    result: str
    screenshot_b64: str = ""


class SessionRecorder:
    """Accumulates per-step records and writes the JSON + text report."""

    def __init__(self, task: str, start_url: str) -> None:
        self.task = task
        self.start_url = start_url
        self.started = datetime.now()
        self.steps: list[_Step] = []
        self.low_confidence: list[dict[str, Any]] = []

    def add(
        self,
        *,
        step: int,
        url: str,
        screenshot: bytes | None,
        action: dict[str, Any],
        confidence: int,
        result: str,
    ) -> None:
        b64 = base64.b64encode(screenshot).decode("ascii") if screenshot else ""
        self.steps.append(
            _Step(
                step=step,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                url=url,
                action=action,
                confidence=confidence,
                result=result,
                screenshot_b64=b64,
            )
        )
        if confidence < 50:
            self.low_confidence.append(
                {"step": step, "confidence": confidence, "action": action}
            )

    def save(
        self, *, success: bool, final_result: str, outdir: str
    ) -> tuple[str, str]:
        """Write ``session_*.json`` and ``session_*_report.txt``; return paths."""
        ensure_dir(outdir)
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(outdir, f"session_{stamp}.json")
        txt_path = os.path.join(outdir, f"session_{stamp}_report.txt")

        payload = {
            "task": self.task,
            "start_url": self.start_url,
            "started": self.started.isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"),
            "success": success,
            "total_steps": len(self.steps),
            "final_result": final_result,
            "low_confidence_warnings": self.low_confidence,
            "steps": [asdict(s) for s in self.steps],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        lines = [
            "=" * 70,
            "BROWSER AGENT SESSION REPORT",
            "=" * 70,
            f"Task          : {self.task}",
            f"Start URL     : {self.start_url}",
            f"Started       : {payload['started']}",
            f"Finished      : {payload['finished']}",
            f"Outcome       : {'SUCCESS' if success else 'FAILED'}",
            f"Total steps   : {len(self.steps)}",
            "",
            f"Low-confidence actions ({len(self.low_confidence)}):",
        ]
        if self.low_confidence:
            for w in self.low_confidence:
                lines.append(
                    f"  - step {w['step']} (conf {w['confidence']}): {w['action']}"
                )
        else:
            lines.append("  (none)")
        lines += ["", "Step-by-step log:"]
        for s in self.steps:
            lines.append(
                f"  [{s.step:>2}] {s.timestamp} {s.url}\n"
                f"       action={s.action} conf={s.confidence}\n"
                f"       result={s.result}"
            )
        lines += ["", "FINAL RESULT:", final_result, "=" * 70]
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        return json_path, txt_path


async def replay(path: str, log: Callable[[str], None] = print) -> None:
    """Re-execute a saved session visually, 1s between actions."""
    from actions import Action, execute_action
    from browser import BrowserController

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    log(f"Replaying: {path}")
    log(f"Task : {data.get('task')}")
    log(f"Steps: {data.get('total_steps')}\n")

    browser = BrowserController()
    try:
        await browser.start()
        await browser.navigate(data.get("start_url", "https://www.google.com"))
        for s in data.get("steps", []):
            act = s.get("action") or {}
            action = Action(
                action_type=act.get("action", "done"),
                params=act.get("params", {}),
                reason=act.get("reason", ""),
                confidence=int(s.get("confidence", 100)),
            )
            log(f"[{s.get('step')}] {action}  ({action.reason})")
            try:
                result = await execute_action(browser, action)
            except Exception as exc:  
                log(f"   [skip] {exc}")
                result = None
            if result is not None:
                log(f"\nDONE — {result}")
                break
            await asyncio.sleep(1.0)
    finally:
        await browser.stop()
