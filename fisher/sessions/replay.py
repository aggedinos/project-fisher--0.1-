"""Read-only access to recorded sessions.

Replay returns recorded events for UI playback or inspection. It deliberately
does not import a browser controller or execute a tool call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fisher.models import AgentEvent


_SESSION_ID = re.compile(r"session-\d{6,}\Z")


def get_session(root: str | Path, session_id: str) -> dict[str, Any]:
    """Load one complete session snapshot by its generated ID."""
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("Invalid session ID")
    directory = Path(root) / session_id
    if directory.is_symlink():
        raise ValueError("Session directory must not be a symlink")
    snapshot = directory / "session.json"
    if snapshot.is_symlink():
        raise ValueError("Session record must not be a symlink")
    with snapshot.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("id") != session_id:
        raise ValueError("Invalid session record")
    return payload


def list_sessions(root: str | Path) -> list[dict[str, Any]]:
    """Return lightweight summaries, newest generated ID first."""
    directory = Path(root)
    if not directory.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for item in sorted(directory.iterdir(), key=lambda entry: entry.name, reverse=True):
        if not item.is_dir() or item.is_symlink() or not _SESSION_ID.fullmatch(item.name):
            continue
        try:
            payload = get_session(directory, item.name)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        summaries.append(
            {
                key: payload.get(key)
                for key in (
                    "id",
                    "task",
                    "provider",
                    "started_at",
                    "finished_at",
                    "status",
                    "answer",
                )
            }
            | {"step_count": len(payload.get("steps", []))}
        )
    return summaries


def replay_events(root: str | Path, session_id: str) -> list[AgentEvent]:
    """Play back stored event values without re-executing browser actions."""
    payload = get_session(root, session_id)
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Invalid session events")
    if events:
        return [AgentEvent.model_validate(event) for event in events]
    # Older or step-only records can still be inspected in chronological order.
    return [
        AgentEvent(
            type="ToolFinished",
            task_id=session_id,
            timestamp=step["timestamp"],
            data={"recorded_step": step},
        )
        for step in payload.get("steps", [])
    ]
