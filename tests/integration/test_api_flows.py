"""Exercise task control and live events through the local HTTP API.

The provider is deterministic; the API, orchestrator, browser, and fixture site
are real.  unittest is used so pytest can discover these tests as well.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx
import uvicorn
from playwright.async_api import async_playwright

from fisher.api import create_api
from fisher.app import FisherApplication
from fisher.config.settings import Settings
from fisher.models import ModelDecision, Observation, PlanStep, TaskPlan, ToolCall
from fisher.providers.base import ProviderError


FIXTURES = Path(__file__).parent / "fixtures"
BROWSER_EXECUTABLE = (
    os.environ.get("FISHER_TEST_BROWSER")
    or os.environ.get("FISHER_EXECUTABLE_PATH")
    or next(
        (path for name in ("chromium", "chromium-browser", "google-chrome", "chrome")
         if (path := shutil.which(name))),
        None,
    )
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def _element_id(observation: Observation, name: str) -> str:
    matches = [item.id for item in observation.elements if item.name == name]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {name!r} control; found {len(matches)}")
    return matches[0]


class _WaitingProvider:
    label = "fixture-waiting"

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        return TaskPlan(goal=task, steps=[PlanStep(id="wait", description="Wait for stop")])

    async def decide(self, *args: object, **kwargs: object) -> ModelDecision:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("The waiting provider should have been cancelled")


class _FormProvider:
    label = "fixture-form"

    def __init__(self, *, stop_after_denial: bool = False) -> None:
        self.stop_after_denial = stop_after_denial
        self.submission_attempted = False

    async def plan(self, task: str, observation: Observation) -> TaskPlan:
        return TaskPlan(
            goal=task,
            steps=[
                PlanStep(id="fill", description="Fill the order form"),
                PlanStep(id="submit", description="Submit the order with approval"),
            ],
        )

    async def decide(
        self, task: str, plan: TaskPlan, observation: Observation,
        memory: str, tools: object, image: bytes | None = None,
    ) -> ModelDecision:
        if "Order submitted for Taylor Fisher" in observation.text:
            return ModelDecision(answer="The order was submitted for Taylor Fisher.")
        if self.stop_after_denial and self.submission_attempted:
            return ModelDecision(answer="The order was not submitted.")
        for name, value in (
            ("Full name", "Taylor Fisher"),
            ("Email address", "taylor@example.test"),
        ):
            field = next(item for item in observation.elements if item.name == name)
            if field.value != value:
                return ModelDecision(
                    tool_call=ToolCall(
                        name="fill_element", arguments={"element_id": field.id, "text": value}
                    )
                )
        self.submission_attempted = True
        return ModelDecision(
            tool_call=ToolCall(
                name="click_element",
                arguments={"element_id": _element_id(observation, "Place order")},
            ),
            completed_step_ids=["fill", "submit"],
        )


async def _next_sse_event(lines: object) -> tuple[int, dict[str, object]]:
    event_id: int | None = None
    data: str | None = None
    async for line in lines:
        if line.startswith("id: "):
            event_id = int(line[4:])
        elif line.startswith("data: "):
            data = line[6:]
        elif line == "" and data is not None:
            if event_id is None:
                raise AssertionError("SSE data had no event ID")
            return event_id, json.loads(data)
    raise AssertionError("SSE stream closed before the expected event")


class ApiFlowTests(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        handler = partial(_QuietHandler, directory=str(FIXTURES))
        cls.site = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.site_thread = Thread(target=cls.site.serve_forever, daemon=True)
        cls.site_thread.start()
        cls.site_url = f"http://127.0.0.1:{cls.site.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.site.shutdown()
        cls.site.server_close()
        cls.site_thread.join(timeout=2)

    async def asyncSetUp(self) -> None:
        self.data_dir = tempfile.TemporaryDirectory(prefix="fisher-api-test-")
        settings = Settings(
            data_dir=Path(self.data_dir.name), allow_private_network=True,
            executable_path=Path(BROWSER_EXECUTABLE) if BROWSER_EXECUTABLE else None,
            max_steps=8,
        )
        self.service = FisherApplication(settings)
        self.provider: object | None = None
        self.provider_patch = patch("fisher.providers.build_provider", side_effect=lambda _: self.provider)
        self.provider_patch.start()
        self.api = create_api(self.service)

        self.api_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.api_socket.bind(("127.0.0.1", 0))
        self.api_socket.listen(128)
        port = self.api_socket.getsockname()[1]
        config = uvicorn.Config(
            self.api, host="127.0.0.1", port=port, log_level="error",
            access_log=False, lifespan="off",
        )
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.api_socket]))

        async def server_ready() -> None:
            while not self.server.started:
                if self.server_task.done():
                    raise AssertionError("The API server stopped during startup")
                await asyncio.sleep(0.01)

        await asyncio.wait_for(server_ready(), timeout=5)
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=15)

    async def asyncTearDown(self) -> None:
        status = self.service.status()
        if status["running"] and status["task_id"]:
            await self.service.stop_task(status["task_id"])
        await self.client.aclose()
        self.server.should_exit = True
        await asyncio.wait_for(self.server_task, timeout=5)
        self.api_socket.close()
        self.provider_patch.stop()
        self.data_dir.cleanup()

    async def _start(self, task: str, path: str) -> str:
        response = await self.client.post(
            "/api/tasks", json={"task": task, "url": f"{self.site_url}/{path}"}
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["task_id"]

    async def _until(self, lines: object, event_type: str) -> tuple[int, dict[str, object]]:
        while True:
            event_id, event = await asyncio.wait_for(_next_sse_event(lines), timeout=15)
            if event["type"] == event_type:
                return event_id, event

    async def test_start_stop_and_live_sse(self) -> None:
        self.provider = _WaitingProvider()
        task_id = await self._start("Wait until stopped", "product.html")
        await asyncio.wait_for(self.provider.entered.wait(), timeout=10)

        status = (await self.client.get("/api/status")).json()
        self.assertTrue(status["running"])
        self.assertEqual(status["task_id"], task_id)
        busy = await self.client.post(
            "/api/tasks", json={"task": "Another task", "url": f"{self.site_url}/index.html"}
        )
        self.assertEqual(busy.status_code, 409)

        async with self.client.stream("GET", f"/api/tasks/{task_id}/events") as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            lines = response.aiter_lines()
            first_id, first = await self._until(lines, "agent_started")
            self.assertEqual(first["task_id"], task_id)
            stopped = await self.client.post(f"/api/tasks/{task_id}/stop")
            self.assertEqual(stopped.status_code, 200, stopped.text)
            self.assertEqual(stopped.json(), {"status": "stopped"})
            final_id, final = await self._until(lines, "agent_finished")
            self.assertGreater(final_id, first_id)
            self.assertEqual(final["data"]["result"]["status"], "stopped")

        status = (await self.client.get("/api/status")).json()
        self.assertFalse(status["running"])
        self.assertEqual(status["result"]["status"], "stopped")

    async def test_permission_approval_over_http_completes_form(self) -> None:
        self.provider = _FormProvider()
        task_id = await self._start("Submit the order after approval", "form.html")

        async with self.client.stream("GET", f"/api/tasks/{task_id}/events") as response:
            self.assertEqual(response.status_code, 200)
            lines = response.aiter_lines()
            request_id_seq, event = await self._until(lines, "permission_requested")
            request = event["data"]["request"]
            self.assertEqual(request["call"]["name"], "click_element")
            self.assertEqual(request["risk"], "high")
            self.assertEqual(event["task_id"], task_id)
            self.assertTrue((await self.client.get("/api/status")).json()["running"])

            frame = await self.client.get(f"/api/tasks/{task_id}/frame")
            self.assertEqual(frame.status_code, 200, frame.text[:200])
            self.assertEqual(frame.headers["content-type"], "image/jpeg")
            self.assertTrue(frame.content.startswith(b"\xff\xd8"))

            approved = await self.client.post(
                f"/api/tasks/{task_id}/permissions/{request['id']}", json={"approved": True}
            )
            self.assertEqual(approved.status_code, 200, approved.text)
            self.assertEqual(approved.json(), {"accepted": True})
            final_id, final = await self._until(lines, "agent_finished")
            self.assertGreater(final_id, request_id_seq)
            self.assertEqual(final["data"]["result"]["status"], "completed")
            self.assertIn("submitted", final["data"]["result"]["answer"])

        repeated = await self.client.post(
            f"/api/tasks/{task_id}/permissions/{request['id']}", json={"approved": True}
        )
        self.assertEqual(repeated.status_code, 404)

    async def test_permission_denial_does_not_submit(self) -> None:
        self.provider = _FormProvider(stop_after_denial=True)
        task_id = await self._start("Fill the form and request submission", "form.html")

        async with self.client.stream("GET", f"/api/tasks/{task_id}/events") as response:
            self.assertEqual(response.status_code, 200)
            lines = response.aiter_lines()
            _, event = await self._until(lines, "permission_requested")
            request_id = event["data"]["request"]["id"]
            denied = await self.client.post(
                f"/api/tasks/{task_id}/permissions/{request_id}", json={"approved": False}
            )
            self.assertEqual(denied.status_code, 200, denied.text)
            _, result_event = await self._until(lines, "tool_result")
            self.assertEqual(result_event["data"]["error_code"], "permission_required")
            _, final = await self._until(lines, "agent_finished")
            self.assertEqual(final["data"]["result"]["status"], "blocked")
            self.assertIn("not approved", final["data"]["result"]["answer"])

    async def test_unknown_task_and_permission_return_not_found(self) -> None:
        self.assertEqual((await self.client.post("/api/tasks/missing/stop")).status_code, 404)
        self.assertEqual((await self.client.get("/api/tasks/missing/events")).status_code, 404)
        self.assertEqual(
            (await self.client.post(
                "/api/tasks/missing/permissions/missing", json={"approved": True}
            )).status_code,
            404,
        )

    async def test_cross_origin_request_is_rejected(self) -> None:
        same_origin = str(self.client.base_url).rstrip("/")
        accepted = await self.client.get("/api/health", headers={"Origin": same_origin})
        self.assertEqual(accepted.status_code, 200)
        port = self.client.base_url.port
        rejected = await self.client.post(
            "/api/tasks", json={"task": "Read page"},
            headers={"Origin": f"http://127.0.0.1:{port + 1}"},
        )
        self.assertEqual(rejected.status_code, 403)
        malformed = await self.client.get("/api/health", headers={"Origin": "http://127.0.0.1:bad"})
        self.assertEqual(malformed.status_code, 403)

    async def test_startup_error_reports_safe_message(self) -> None:
        with patch("fisher.providers.build_provider", side_effect=RuntimeError("token=secret123")):
            task_id = await self._start("Read page", "product.html")
            runtime = self.service.get_task(task_id)
            await asyncio.wait_for(runtime.done.wait(), timeout=5)
            self.assertEqual(runtime.result.status, "error")
            self.assertNotIn("secret123", json.dumps(runtime.result.model_dump()))
            self.assertTrue(any(event.type == "agent_finished" for _, event in runtime.events))
        with patch("fisher.providers.build_provider", side_effect=ProviderError("GEMINI_API_KEY is required")):
            task_id = await self._start("Read page", "product.html")
            runtime = self.service.get_task(task_id)
            await asyncio.wait_for(runtime.done.wait(), timeout=5)
            self.assertIn("GEMINI_API_KEY is required", runtime.result.error)

    async def test_built_control_center_runs_and_stops_a_task(self) -> None:
        self.provider = _WaitingProvider()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, executable_path=BROWSER_EXECUTABLE
            )
            try:
                page = await browser.new_page()
                await page.goto(str(self.client.base_url), wait_until="networkidle")
                await page.get_by_role("heading", name="Control center.").wait_for()
                self.assertIn("Ollama", await page.locator(".composer-model").inner_text())
                await page.get_by_role("textbox", name="Task", exact=True).fill("Wait until stopped")
                await page.get_by_role("textbox", name="Starting URL").fill(
                    f"{self.site_url}/product.html"
                )
                await page.get_by_role("button", name="Run task").click()
                await asyncio.wait_for(self.provider.entered.wait(), timeout=10)
                await page.get_by_role("button", name="Stop", exact=True).click()
                await page.get_by_text("Stopped", exact=True).first.wait_for(timeout=10_000)
            finally:
                await browser.close()

    async def test_control_center_approves_form_submission(self) -> None:
        self.provider = _FormProvider()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, executable_path=BROWSER_EXECUTABLE
            )
            try:
                page = await browser.new_page()
                await page.goto(str(self.client.base_url), wait_until="networkidle")
                await page.get_by_role("textbox", name="Task", exact=True).fill(
                    "Submit the order after approval"
                )
                await page.get_by_role("textbox", name="Starting URL").fill(
                    f"{self.site_url}/form.html"
                )
                await page.get_by_role("button", name="Run task").click()
                dialog = page.get_by_role("alertdialog", name="Permission request")
                await dialog.wait_for(timeout=20_000)
                self.assertIn("HIGH RISK", await dialog.inner_text())
                await dialog.get_by_role("button", name="Approve").click()
                await page.get_by_text("The order was submitted for Taylor Fisher.").first.wait_for(
                    timeout=20_000
                )
            finally:
                await browser.close()

    async def test_control_center_settings_persist_without_keys(self) -> None:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, executable_path=BROWSER_EXECUTABLE
            )
            try:
                page = await browser.new_page()
                await page.goto(str(self.client.base_url), wait_until="networkidle")
                await page.locator(".topbar-settings").click()
                dialog = page.get_by_role("dialog", name="Agent settings")
                self.assertIn("Saved on this device", await dialog.inner_text())
                await dialog.locator(".provider-card").filter(has_text="Ollama").click()
                await dialog.get_by_role("button", name="Supervised").click()
                await dialog.get_by_label("Profile").select_option("persistent")
                await dialog.locator("#headless-toggle").uncheck()
                await dialog.get_by_role("button", name="Save settings").click()
                await dialog.wait_for(state="hidden")
                saved = (await self.client.get("/api/settings")).json()
                self.assertEqual(saved["provider"], "ollama")
                self.assertEqual(saved["model"], "llama3.2-vision")
                self.assertEqual(saved["permission_mode"], "supervised")
                self.assertEqual(saved["profile"], "persistent")
                self.assertFalse(saved["headless"])
                self.assertNotIn("api_key", json.dumps(saved).lower())
                restored = FisherApplication(Settings(data_dir=Path(self.data_dir.name)))
                self.assertEqual(restored.public_settings()["provider"], "ollama")
                self.assertEqual(restored.public_settings()["permission_mode"], "supervised")
            finally:
                await browser.close()


if __name__ == "__main__":
    import unittest

    unittest.main()
