"""Shared application service used by the CLI and the local GUI API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from fisher.models import AgentEvent, PermissionRequest, RunResult

logger = logging.getLogger(__name__)


class TaskBusyError(RuntimeError):
    """A browser task is already active."""


class UnknownTaskError(KeyError):
    """The requested task is not retained by this process."""


class TaskRuntime:
    def __init__(self, task_id: str, task: str, url: str) -> None:
        self.id = task_id
        self.task = task
        self.url = url
        self.browser: Any = None
        self.orchestrator: Any = None
        self.worker: asyncio.Task[None] | None = None
        self.events: deque[tuple[int, AgentEvent]] = deque(maxlen=1000)
        self.subscribers: set[asyncio.Queue[tuple[int, AgentEvent]]] = set()
        self.permissions: dict[str, asyncio.Future[bool]] = {}
        self.frame: bytes | None = None
        self.result: RunResult | None = None
        self.done = asyncio.Event()
        self._sequence = 0

    def publish(self, event: AgentEvent) -> None:
        self._sequence += 1
        entry = (self._sequence, event)
        self.events.append(entry)
        for subscriber in tuple(self.subscribers):
            try:
                subscriber.put_nowait(entry)
            except asyncio.QueueFull:
                logger.warning("Dropping GUI event for a slow subscriber")

    def subscribe(self) -> asyncio.Queue[tuple[int, AgentEvent]]:
        queue: asyncio.Queue[tuple[int, AgentEvent]] = asyncio.Queue(maxsize=256)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[tuple[int, AgentEvent]]) -> None:
        self.subscribers.discard(queue)

    async def request_permission(self, request: PermissionRequest) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.permissions[request.id] = future
        self.publish(
            AgentEvent(
                type="permission_requested", task_id=self.id, data={"request": request.model_dump()}
            )
        )
        try:
            return await future
        finally:
            self.permissions.pop(request.id, None)

    def resolve_permission(self, request_id: str, approved: bool) -> bool:
        future = self.permissions.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def cancel_permissions(self) -> None:
        for future in self.permissions.values():
            if not future.done():
                future.cancel()


class FisherApplication:
    """Owns one active task and its events without exposing provider secrets."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._tasks: dict[str, TaskRuntime] = {}
        self._current_id: str | None = None
        self._settings_path = Path(settings.data_dir) / "settings.json"
        self._load_saved_settings()

    def _load_saved_settings(self) -> None:
        if not self._settings_path.exists():
            return
        try:
            saved = json.loads(self._settings_path.read_text(encoding="utf-8"))
            allowed = {"provider", "model", "permission_mode", "profile", "headless"}
            process_overrides = {
                "provider": "FISHER_PROVIDER",
                "model": "FISHER_MODEL",
                "permission_mode": "FISHER_PERMISSION_MODE",
                "profile": "FISHER_PROFILE",
                "headless": "FISHER_HEADLESS",
            }
            explicit = {
                name for name, variable in process_overrides.items() if variable in os.environ
            }
            if "FISHER_PROFILE_MODE" in os.environ:
                explicit.add("profile")
            if "provider" in explicit:
                explicit.add("model")
            self.settings = self.settings.with_overrides(
                **{key: value for key, value in saved.items() if key in allowed - explicit}
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid saved settings: %s", exc)

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {"provider", "model", "permission_mode", "profile", "headless"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        updated = self.settings.with_overrides(**values)
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        safe = {key: getattr(updated, key) for key in allowed}
        safe["permission_mode"] = str(
            getattr(safe["permission_mode"], "value", safe["permission_mode"])
        )
        temporary = self._settings_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(safe, indent=2), encoding="utf-8")
        temporary.replace(self._settings_path)
        self.settings = updated
        return self.public_settings()

    def public_settings(self) -> dict[str, Any]:
        settings = self.settings
        mode = settings.permission_mode
        return {
            "provider": settings.provider,
            "model": settings.model,
            "permission_mode": str(getattr(mode, "value", mode)),
            "profile": settings.profile,
            "headless": settings.headless,
            "providers": [
                {
                    "name": "gemini",
                    "configured": bool(settings.gemini_api_key.get_secret_value()),
                    "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
                },
                {
                    "name": "ollama",
                    "configured": True,
                    "models": ["llama3.2-vision", "qwen2.5"],
                },
                {
                    "name": "nvidia",
                    "configured": bool(settings.nvidia_api_key.get_secret_value()),
                    "models": ["meta/llama-3.2-90b-vision-instruct"],
                },
            ],
        }

    def get_task(self, task_id: str) -> TaskRuntime:
        runtime = self._tasks.get(task_id)
        if runtime is None:
            raise UnknownTaskError(task_id)
        return runtime

    def status(self) -> dict[str, Any]:
        runtime = self._tasks.get(self._current_id or "")
        return {
            "running": bool(runtime and not runtime.done.is_set()),
            "task_id": runtime.id if runtime else None,
            "task": runtime.task if runtime else None,
            "result": runtime.result.model_dump() if runtime and runtime.result else None,
        }

    async def start_task(self, task: str, url: str, **overrides: Any) -> TaskRuntime:
        task = task.strip()
        if not task:
            raise ValueError("Task must not be empty")
        async with self._lock:
            current = self._tasks.get(self._current_id or "")
            if current and not current.done.is_set():
                raise TaskBusyError("Another task is running")
            settings = self.settings.with_overrides(
                **{key: value for key, value in overrides.items() if value is not None}
            )
            required_key = {"gemini": "GEMINI_API_KEY", "nvidia": "NVIDIA_API_KEY"}.get(
                settings.provider
            )
            if required_key and not settings.api_key_for_provider():
                raise ValueError(
                    f"{required_key} is required. Add it to your local .env and restart Fisher."
                )
            runtime = TaskRuntime(str(uuid4()), task, url)
            self._tasks[runtime.id] = runtime
            self._current_id = runtime.id
            runtime.worker = asyncio.create_task(self._run(runtime, settings))
            return runtime

    async def _run(self, runtime: TaskRuntime, settings: Any) -> None:
        from fisher.agent.orchestrator import AgentOrchestrator
        from fisher.browser.controller import BrowserController
        from fisher.providers import build_provider
        from fisher.providers.base import ProviderError
        from fisher.sessions.recorder import SessionRecorder
        from fisher.tools.registry import ToolRegistry

        browser = BrowserController(
            allow_private_network=settings.allow_private_network, data_dir=settings.data_dir
        )
        runtime.browser = browser

        async def on_event(event: AgentEvent) -> None:
            runtime.publish(event)
            if event.type == "observation_updated":
                try:
                    runtime.frame = await browser.screenshot()
                    runtime.publish(AgentEvent(type="frame", task_id=runtime.id))
                except Exception as exc:
                    logger.debug("Preview frame unavailable (%s)", type(exc).__name__)

        try:
            provider = build_provider(settings)
            await browser.start(
                profile=settings.profile,
                headless=settings.headless,
                executable_path=settings.executable_path,
            )
            registry = ToolRegistry(browser)
            recorder = SessionRecorder(root=Path(settings.data_dir) / "sessions")
            orchestrator = AgentOrchestrator(
                provider=provider,
                browser=browser,
                registry=registry,
                permission_mode=settings.permission_mode,
                approve=runtime.request_permission,
                on_event=on_event,
                recorder=recorder,
                max_steps=settings.max_steps,
            )
            runtime.orchestrator = orchestrator
            runtime.result = await orchestrator.run(runtime.task, runtime.url, task_id=runtime.id)
            if not any(event.type == "agent_finished" for _, event in runtime.events):
                runtime.publish(
                    AgentEvent(
                        type="agent_finished",
                        task_id=runtime.id,
                        data={"result": runtime.result.model_dump()},
                    )
                )
        except asyncio.CancelledError:
            runtime.result = RunResult(
                task_id=runtime.id, status="stopped", error="Cancelled by user"
            )
            runtime.publish(
                AgentEvent(
                    type="agent_finished",
                    task_id=runtime.id,
                    data={"result": runtime.result.model_dump()},
                )
            )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, ProviderError)
                else f"Task failed ({type(exc).__name__})"
            )
            logger.error("Task %s failed (%s): %s", runtime.id, type(exc).__name__, message)
            runtime.result = RunResult(task_id=runtime.id, status="error", error=message)
            runtime.publish(
                AgentEvent(type="agent_error", task_id=runtime.id, data={"message": message})
            )
            runtime.publish(
                AgentEvent(
                    type="agent_finished",
                    task_id=runtime.id,
                    data={"result": runtime.result.model_dump()},
                )
            )
        finally:
            runtime.cancel_permissions()
            try:
                await browser.close()
            except Exception as exc:
                logger.error("Browser cleanup failed (%s)", type(exc).__name__)
            runtime.done.set()

    async def stop_task(self, task_id: str) -> None:
        runtime = self.get_task(task_id)
        if runtime.done.is_set():
            return
        if runtime.orchestrator is not None:
            runtime.orchestrator.cancel()
        runtime.cancel_permissions()
        if runtime.worker is not None and runtime.orchestrator is None:
            runtime.worker.cancel()
        if runtime.worker is not None:
            try:
                await asyncio.wait_for(runtime.done.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for task %s to stop", task_id)

    async def run_once(
        self,
        task: str,
        url: str,
        *,
        on_event: Any = None,
        approve: Any = None,
        **overrides: Any,
    ) -> RunResult:
        """CLI entry point using the same browser, registry and orchestrator as the GUI."""
        from fisher.agent.orchestrator import AgentOrchestrator
        from fisher.browser.controller import BrowserController
        from fisher.providers import build_provider
        from fisher.sessions.recorder import SessionRecorder
        from fisher.tools.registry import ToolRegistry

        settings = self.settings.with_overrides(
            **{k: v for k, v in overrides.items() if v is not None}
        )
        browser = BrowserController(
            allow_private_network=settings.allow_private_network, data_dir=settings.data_dir
        )
        try:
            provider = build_provider(settings)
            await browser.start(
                profile=settings.profile,
                headless=settings.headless,
                executable_path=settings.executable_path,
            )
            orchestrator = AgentOrchestrator(
                provider=provider,
                browser=browser,
                registry=ToolRegistry(browser),
                permission_mode=settings.permission_mode,
                approve=approve,
                on_event=on_event,
                recorder=SessionRecorder(root=Path(settings.data_dir) / "sessions"),
                max_steps=settings.max_steps,
            )
            return await orchestrator.run(task, url)
        finally:
            await browser.close()
