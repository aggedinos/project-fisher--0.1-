"""Loopback-only HTTP contract for the React control center."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from fisher.app import FisherApplication, TaskBusyError, UnknownTaskError
from fisher.config.settings import load_settings
from fisher.sessions.recorder import SessionRecorder


class StartTaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=10_000)
    url: str = Field(default="https://duckduckgo.com", max_length=2048)
    provider: str | None = None
    model: str | None = None
    permission_mode: str | None = None
    profile: str | None = None
    headless: bool | None = None


class SettingsRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    permission_mode: str | None = None
    profile: str | None = None
    headless: bool | None = None


class PermissionResponse(BaseModel):
    approved: bool


def create_api(application: FisherApplication | None = None) -> FastAPI:
    service = application or FisherApplication(load_settings())
    api = FastAPI(title="Project Fisher", version="0.2.0", docs_url=None, redoc_url=None)
    api.state.service = service

    @api.middleware("http")
    async def loopback_only(request: Request, call_next: Any) -> Response:
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return Response("Fisher only accepts loopback hosts", status_code=403)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            try:
                same_origin = (
                    parsed.scheme == request.url.scheme
                    and parsed.hostname == host
                    and (parsed.port or (443 if parsed.scheme == "https" else 80))
                    == (request.url.port or (443 if request.url.scheme == "https" else 80))
                    and parsed.username is None
                    and parsed.password is None
                    and parsed.path in {"", "/"}
                    and not parsed.query
                    and not parsed.fragment
                )
            except ValueError:
                same_origin = False
            if not same_origin:
                return Response("Cross-origin request rejected", status_code=403)
        return await call_next(request)

    @api.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "healthy", "running": service.status()["running"]}

    @api.get("/api/status")
    async def status() -> dict[str, Any]:
        return service.status()

    @api.get("/api/settings")
    async def settings() -> dict[str, Any]:
        return service.public_settings()

    @api.put("/api/settings")
    async def update_settings(payload: SettingsRequest) -> dict[str, Any]:
        try:
            return service.update_settings(payload.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @api.post("/api/tasks", status_code=202)
    async def start_task(payload: StartTaskRequest) -> dict[str, str]:
        overrides = payload.model_dump(exclude={"task", "url"}, exclude_none=True)
        try:
            runtime = await service.start_task(payload.task, payload.url, **overrides)
        except TaskBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"task_id": runtime.id}

    @api.post("/api/tasks/{task_id}/stop")
    async def stop_task(task_id: str) -> dict[str, str]:
        try:
            await service.stop_task(task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="Unknown task") from exc
        return {"status": "stopped"}

    @api.post("/api/tasks/{task_id}/permissions/{request_id}")
    async def resolve_permission(
        task_id: str, request_id: str, payload: PermissionResponse
    ) -> dict[str, bool]:
        try:
            runtime = service.get_task(task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="Unknown task") from exc
        if not runtime.resolve_permission(request_id, payload.approved):
            raise HTTPException(status_code=404, detail="Unknown permission request")
        return {"accepted": True}

    @api.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str) -> StreamingResponse:
        try:
            runtime = service.get_task(task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="Unknown task") from exc

        async def stream() -> Any:
            queue = runtime.subscribe()
            last_seen = 0
            try:
                for sequence, event in list(runtime.events):
                    last_seen = sequence
                    yield f"id: {sequence}\ndata: {event.model_dump_json()}\n\n"
                while not runtime.done.is_set() or not queue.empty():
                    try:
                        sequence, event = await asyncio.wait_for(queue.get(), timeout=10)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if sequence <= last_seen:
                        continue
                    last_seen = sequence
                    yield f"id: {sequence}\ndata: {event.model_dump_json()}\n\n"
            finally:
                runtime.unsubscribe(queue)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @api.get("/api/tasks/{task_id}/frame")
    async def frame(task_id: str) -> Response:
        try:
            runtime = service.get_task(task_id)
        except UnknownTaskError as exc:
            raise HTTPException(status_code=404, detail="Unknown task") from exc
        if runtime.frame is None:
            raise HTTPException(status_code=404, detail="No browser frame yet")
        return Response(
            runtime.frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    def recorder() -> SessionRecorder:
        return SessionRecorder(root=Path(service.settings.data_dir) / "sessions")

    @api.get("/api/sessions")
    async def sessions() -> Any:
        return recorder().list_sessions()

    @api.get("/api/sessions/{session_id}")
    async def session(session_id: str) -> Any:
        try:
            return recorder().get_session(session_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

    web_dir = Path(__file__).resolve().parent.parent / "web"

    @api.get("/{asset_path:path}")
    async def static_content(asset_path: str) -> Response:
        if asset_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Unknown API route")
        root = web_dir.resolve()
        candidate = (root / asset_path).resolve()
        if asset_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        index = root / "index.html"
        if index.is_file() and (not asset_path or "." not in Path(asset_path).name):
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend is not built")

    return api
