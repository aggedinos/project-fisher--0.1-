"""Session history preserves useful diagnostics without replaying side effects."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fisher.models import (
    AgentEvent,
    ElementInfo,
    Observation,
    PlanStep,
    TaskPlan,
    ToolCall,
    ToolResult,
)
from fisher.sessions import SessionRecorder, get_session, list_sessions, replay_events
from fisher.sessions import recorder as recorder_module


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_recording_redacts_inputs_and_keeps_screenshots_separate(self) -> None:
        recorder = SessionRecorder(self.root)
        session_id = recorder.start("Log in", "ollama")
        self.assertEqual(session_id, "session-000001")

        call = ToolCall(
            name="browser_fill", arguments={"selector": "#password", "text": "hunter2-secret"}
        )
        recorder.record_event(
            AgentEvent(type="ToolStarted", task_id="task-1", data={"call": call.model_dump()})
        )
        observation = Observation(
            url="https://example.test/login?token=abc123#private",
            title="Sign in",
            text="Authorization: Bearer abc.def.ghi",
            elements=[
                ElementInfo(id="pwd", role="textbox", value="hunter2-secret", input_type="password")
            ],
        )
        result = ToolResult(
            call=call,
            success=True,
            changed=True,
            message="Entered hunter2-secret",
            after_state=observation,
            data={"verification": "Value hunter2-secret accepted", "confidence": 0.8},
        )
        plan = TaskPlan(goal="Log in", steps=[PlanStep(id="1", description="Fill password")])
        recorder.record_step(call, result, plan)

        image = b"\x89PNG\r\n\x1a\n" + b"pixel bytes"
        click = ToolCall(name="browser_click", arguments={"selector": "#continue"})
        recorder.record_step(click, ToolResult(call=click, success=True), screenshot=image)
        recorder.finish("completed", "Done")

        session = get_session(self.root, session_id)
        raw_json = (self.root / session_id / "session.json").read_text(encoding="utf-8")
        for secret in ("hunter2-secret", "abc123", "abc.def.ghi"):
            self.assertNotIn(secret, raw_json)
        self.assertEqual(session["steps"][0]["call"]["arguments"]["text"], "[REDACTED]")
        self.assertEqual(session["steps"][0]["observation"]["elements"][0]["value"], "[REDACTED]")
        self.assertEqual(
            session["steps"][0]["url"], "https://example.test/login?token=%5BREDACTED%5D"
        )
        self.assertEqual(session["steps"][0]["verification"], "Value [REDACTED] accepted")
        self.assertEqual(session["steps"][0]["confidence"], 0.8)
        screenshot_ref = session["steps"][1]["screenshot"]
        self.assertEqual(screenshot_ref, "screenshots/step-002.png")
        self.assertEqual((self.root / session_id / screenshot_ref).read_bytes(), image)
        self.assertNotIn("pixel bytes", raw_json)
        self.assertEqual(session["status"], "completed")

    def test_sequential_ids_and_read_only_replay(self) -> None:
        first = SessionRecorder(self.root)
        first_id = first.start("Visit page", "gemini")
        call = ToolCall(name="browser_click", arguments={"selector": "#buy"})
        first.record_event(
            AgentEvent(type="ToolFinished", task_id="task-1", data={"call": call.model_dump()})
        )
        first.finish("completed", "Done")
        second = SessionRecorder(self.root)
        second_id = second.start("Other", "ollama")
        second.finish("failed", "No page")

        self.assertEqual((first_id, second_id), ("session-000001", "session-000002"))
        self.assertEqual([item["id"] for item in list_sessions(self.root)], [second_id, first_id])
        before = (self.root / first_id / "session.json").read_bytes()
        events = replay_events(self.root, first_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "ToolFinished")
        self.assertEqual(events[0].data["call"]["name"], "browser_click")
        self.assertEqual((self.root / first_id / "session.json").read_bytes(), before)

    def test_step_only_session_replays_as_recorded_events(self) -> None:
        recorder = SessionRecorder(self.root)
        session_id = recorder.start("Search", "ollama")
        call = ToolCall(name="browser_click", arguments={"selector": "#search"})
        recorder.record_step(call, ToolResult(call=call, success=True))
        recorder.finish("completed")

        events = recorder.replay(session_id)
        self.assertEqual(
            [(event.type, event.data["recorded_step"]["index"]) for event in events],
            [("ToolFinished", 1)],
        )

    def test_input_screenshot_is_omitted(self) -> None:
        recorder = SessionRecorder(self.root)
        session_id = recorder.start("Use password hunter2", "ollama")
        call = ToolCall(name="browser_fill", arguments={"params": {"text": "xy"}})
        recorder.record_step(
            call,
            ToolResult(call=call, success=True, message="Entered xy"),
            screenshot=b"private image",
        )
        recorder.finish("completed")
        saved = get_session(self.root, session_id)
        self.assertIsNone(saved["steps"][0]["screenshot"])
        self.assertEqual(saved["task"], "Use password [REDACTED]")
        self.assertEqual(saved["steps"][0]["call"]["arguments"]["params"]["text"], "[REDACTED]")
        self.assertEqual(saved["steps"][0]["result"]["message"], "Entered [REDACTED]")
        self.assertFalse((self.root / session_id / "screenshots").exists())

    def test_malformed_url_is_not_copied_into_session(self) -> None:
        recorder = SessionRecorder(self.root)
        session_id = recorder.start("Read", "ollama")
        recorder.record_event(
            AgentEvent(
                type="ObservationUpdated", task_id="t", data={"url": "https://[bad?token=secret123"}
            )
        )
        recorder.finish("blocked")
        raw = (self.root / session_id / "session.json").read_text(encoding="utf-8")
        self.assertNotIn("secret123", raw)
        self.assertEqual(
            get_session(self.root, session_id)["events"][0]["data"]["url"], "[invalid URL]"
        )

    def test_invalid_ids_and_atomic_snapshot(self) -> None:
        recorder = SessionRecorder(self.root)
        session_id = recorder.start("Read", "ollama")
        snapshot = self.root / session_id / "session.json"
        original = snapshot.read_bytes()

        with patch.object(
            recorder_module.os, "replace", side_effect=OSError("simulated interrupted save")
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                recorder.record_event(AgentEvent(type="ObservationUpdated", task_id="t"))
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8"))["events"], [])

        with self.assertRaisesRegex(ValueError, "Invalid session ID"):
            get_session(self.root, "../elsewhere")


if __name__ == "__main__":
    unittest.main()
