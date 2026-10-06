"""Small, atomic session records for debugging agent runs.

The recorder never stores an inline screenshot. It also removes likely secrets
from event payloads, tool arguments, browser observations, and result text.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

from fisher.models import AgentEvent, Observation, TaskPlan, ToolCall, ToolResult


_REDACTED = "[REDACTED]"
_OMITTED = "[omitted]"
_ID_PATTERN = re.compile(r"session-(\d{6,})\Z")
_SENSITIVE_KEY = re.compile(
    r"password|passwd|passphrase|secret|token|api.?key|authorization|"
    r"cookie|credential|private.?key|access.?key|security.?code|"
    r"otp|cvv|card.?number|ssn|^pin$|^value$",
    re.IGNORECASE,
)
_BINARY_KEY = re.compile(r"screenshot|image|base64|\bb64\b", re.IGNORECASE)
_INPUT_VALUE_KEYS = frozenset({"text", "value", "keys", "input", "content", "query", "message"})
_ASSIGNMENT = re.compile(
    r"(?i)((?<![\w-])(?:password|passwd|passphrase|secret|token|api[_ -]?key|"
    r"authorization|cookie|credential|otp|cvv)\b(?:\s+is\s+|\s*[:=]\s*|\s+))"
    r"([^\s,;&]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_SHAPES = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _is_input_call(name: str) -> bool:
    name = name.lower()
    return any(part in name for part in ("fill", "type", "paste", "insert", "input", "write", "enter_text"))


def _collect_sensitive(value: Any, found: set[str], *, input_args: bool = False, depth: int = 0) -> None:
    """Remember entered values so echoes in tool results can be removed too."""
    if depth > 12:
        return
    value = _plain(value)
    if isinstance(value, Mapping):
        call_input = _is_input_call(str(value.get("name", ""))) and "arguments" in value
        for key, child in value.items():
            label = str(key).lower()
            if _SENSITIVE_KEY.search(label) or (input_args and label in _INPUT_VALUE_KEYS):
                _collect_scalar_strings(child, found)
            else:
                _collect_sensitive(
                    child,
                    found,
                    input_args=input_args or (call_input and label == "arguments"),
                    depth=depth + 1,
                )
    elif isinstance(value, (list, tuple)):
        for child in value[:100]:
            _collect_sensitive(child, found, input_args=input_args, depth=depth + 1)


def _collect_scalar_strings(value: Any, found: set[str]) -> None:
    value = _plain(value)
    if isinstance(value, str) and 1 <= len(value) <= 500:
        found.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            _collect_scalar_strings(child, found)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_scalar_strings(child, found)


def _scrub_text(value: str, known: set[str], *, url: bool = False, structural: bool = False) -> str:
    value = _BEARER.sub("Bearer " + _REDACTED, value)
    value = _ASSIGNMENT.sub(lambda match: match.group(1) + _REDACTED, value)
    value = _KEY_SHAPES.sub(_REDACTED, value)
    if not structural:
        for secret in sorted(known, key=len, reverse=True):
            if len(secret) < 3:
                value = re.sub(rf"(?<!\w){re.escape(secret)}(?!\w)", _REDACTED, value)
            else:
                value = value.replace(secret, _REDACTED)
    if url and value.startswith(("http://", "https://")):
        try:
            parts = urlsplit(value)
            # Remove URL user information and fragments, which often hold tokens.
            netloc = parts.netloc.rsplit("@", 1)[-1]
            query = urlencode([(key, _REDACTED) for key, _ in parse_qsl(parts.query, keep_blank_values=True)])
            value = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
        except ValueError:
            return "[invalid URL]"
    return value[:4000] + ("…[truncated]" if len(value) > 4000 else "")


def _sanitize(
    value: Any,
    known: set[str],
    *,
    key: str = "",
    input_args: bool = False,
    depth: int = 0,
) -> Any:
    if _SENSITIVE_KEY.search(key) or (input_args and key.lower() in _INPUT_VALUE_KEYS):
        return _REDACTED
    if _BINARY_KEY.search(key):
        return _OMITTED
    if depth > 12:
        return _OMITTED
    value = _plain(value)
    if isinstance(value, Mapping):
        call_input = _is_input_call(str(value.get("name", ""))) and "arguments" in value
        output: dict[str, Any] = {}
        for index, (label, child) in enumerate(value.items()):
            if index >= 100:
                output["_truncated"] = True
                break
            name = str(label)
            output[name] = _sanitize(
                child,
                known,
                key=name,
                input_args=call_input and name == "arguments" or input_args,
                depth=depth + 1,
            )
        return output
    if isinstance(value, (list, tuple)):
        items = [_sanitize(item, known, input_args=input_args, depth=depth + 1) for item in value[:100]]
        if len(value) > 100:
            items.append(_OMITTED)
        return items
    if isinstance(value, (bytes, bytearray)):
        return _OMITTED
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return _scrub_text(str(value), known)
    if isinstance(value, str):
        return _scrub_text(
            value,
            known,
            url=key.lower().endswith("url") or key.lower() == "href",
            structural=key.lower() in {"id", "task_id", "name", "type", "role", "tag", "status"},
        )
    if isinstance(value, float) and not math.isfinite(value):
        return _OMITTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _OMITTED


def _observation_summary(observation: Observation | None, known: set[str]) -> dict[str, Any] | None:
    if observation is None:
        return None
    summary = observation.summary(max_text=1200)
    summary["elements"] = summary["elements"][:50]
    summary["tabs"] = summary["tabs"][:20]
    return _sanitize(summary, known)


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace a complete file, leaving the prior version intact on failure."""
    fd, temporary = tempfile.mkstemp(prefix=".saving-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _image_extension(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "bin"


class SessionRecorder:
    """Persist one run per directory; all methods are synchronous.

    ``start`` returns a deterministic, monotonic ID such as
    ``session-000001``. The same recorder instance handles one run.
    """

    def __init__(self, root: str | Path = "sessions") -> None:
        self.root = Path(root)
        self._session: dict[str, Any] | None = None
        self._path: Path | None = None
        self._known_sensitive: set[str] = set()
        self._lock = threading.RLock()

    @property
    def session_id(self) -> str | None:
        return None if self._session is None else str(self._session["id"])

    def start(self, task: str, provider: str) -> str:
        with self._lock:
            if self._session is not None:
                raise RuntimeError("This recorder already has a session")
            self.root.mkdir(parents=True, exist_ok=True)
            existing = [int(match.group(1)) for item in self.root.iterdir() if (match := _ID_PATTERN.fullmatch(item.name))]
            number = max(existing, default=0) + 1
            while True:
                session_id = f"session-{number:06d}"
                directory = self.root / session_id
                try:
                    directory.mkdir()
                    break
                except FileExistsError:
                    number += 1
            self._path = directory / "session.json"
            self._session = {
                "schema_version": 1,
                "id": session_id,
                "task": _sanitize(task, self._known_sensitive),
                "provider": _sanitize(provider, self._known_sensitive),
                "started_at": _utc_now(),
                "finished_at": None,
                "status": "running",
                "answer": "",
                "events": [],
                "steps": [],
            }
            self._save()
            return session_id

    def _active(self) -> dict[str, Any]:
        if self._session is None or self._path is None:
            raise RuntimeError("Call start() before recording")
        if self._session["finished_at"] is not None:
            raise RuntimeError("Session is already finished")
        return self._session

    def _save(self) -> None:
        assert self._path is not None and self._session is not None
        serialized = json.dumps(self._session, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        _atomic_write(self._path, serialized)

    def record_event(self, event: AgentEvent | Mapping[str, Any]) -> None:
        with self._lock:
            session = self._active()
            payload = _plain(event)
            if not isinstance(payload, Mapping):
                raise TypeError("event must be an AgentEvent or mapping")
            _collect_sensitive(payload, self._known_sensitive)
            clean = _sanitize(payload, self._known_sensitive)
            session["events"].append(clean)
            self._save()

    def record_step(
        self,
        call: ToolCall,
        result: ToolResult,
        plan: TaskPlan | None = None,
        screenshot: bytes | None = None,
    ) -> None:
        with self._lock:
            session = self._active()
            call_data = call.model_dump()
            result_data = result.model_dump(exclude={"before_state", "after_state"})
            _collect_sensitive(call_data, self._known_sensitive)
            _collect_sensitive(result_data, self._known_sensitive)
            observation = result.after_state or result.before_state
            index = len(session["steps"]) + 1
            screenshot_ref: str | None = None
            # Screenshots taken while typing can capture unmasked private input.
            if screenshot is not None and not _is_input_call(call.name):
                if not isinstance(screenshot, bytes):
                    raise TypeError("screenshot must be bytes")
                extension = _image_extension(screenshot)
                screenshot_ref = f"screenshots/step-{index:03d}.{extension}"
                path = self._path.parent / screenshot_ref  # type: ignore[union-attr]
                path.parent.mkdir(exist_ok=True)
                _atomic_write(path, screenshot)
            metadata = result.data if isinstance(result.data, dict) else {}
            step = {
                "index": index,
                "timestamp": _utc_now(),
                "url": _sanitize(observation.url, self._known_sensitive, key="url") if observation else None,
                "observation": _observation_summary(observation, self._known_sensitive),
                "call": _sanitize(call_data, self._known_sensitive),
                "result": _sanitize(result_data, self._known_sensitive),
                "verification": _sanitize(metadata.get("verification"), self._known_sensitive),
                "confidence": _sanitize(metadata.get("confidence", call.arguments.get("confidence")), self._known_sensitive),
                "plan": _sanitize(plan, self._known_sensitive) if plan is not None else None,
                "screenshot": screenshot_ref,
            }
            session["steps"].append(step)
            self._save()

    def finish(self, status: str, answer: str = "") -> dict[str, Any]:
        with self._lock:
            session = self._active()
            session["status"] = _sanitize(status, self._known_sensitive)
            session["answer"] = _sanitize(answer, self._known_sensitive)
            session["finished_at"] = _utc_now()
            self._save()
            return dict(session)

    def list_sessions(self) -> list[dict[str, Any]]:
        from .replay import list_sessions

        return list_sessions(self.root)

    def get_session(self, session_id: str) -> dict[str, Any]:
        from .replay import get_session

        return get_session(self.root, session_id)

    def replay(self, session_id: str) -> list[AgentEvent]:
        from .replay import replay_events

        return replay_events(self.root, session_id)
