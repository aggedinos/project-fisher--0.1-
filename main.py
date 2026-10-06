"""Project Fisher 0.2 command line and local control center entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn

from fisher.api import create_api
from fisher.app import FisherApplication
from fisher.config.settings import load_settings
from fisher.models import AgentEvent, PermissionMode, PermissionRequest
from fisher.providers.base import ProviderError
from fisher.sessions.recorder import SessionRecorder


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Project Fisher 0.2 browser agent")
    command.add_argument("task", nargs="?", help="Task to perform")
    command.add_argument("--url", default="https://duckduckgo.com", help="Starting HTTP(S) URL")
    command.add_argument("--provider", choices=("gemini", "ollama", "nvidia"))
    command.add_argument("--model", help="Model name for the selected provider")
    command.add_argument(
        "--permission-mode", choices=tuple(mode.value for mode in PermissionMode)
    )
    command.add_argument("--profile", choices=("temporary", "persistent"))
    display = command.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--headed", dest="headless", action="store_false")
    command.set_defaults(headless=None)
    command.add_argument("--browser-executable", help="Optional Chromium executable path")
    command.add_argument("--serve", action="store_true", help="Serve the web control center locally")
    command.add_argument("--gui", action="store_true", help="Open the desktop control center")
    command.add_argument("--port", type=int, default=8000, help="Local GUI port (default: 8000)")
    command.add_argument("--replay", metavar="SESSION_ID", help="Read-only session replay")
    command.add_argument("--verbose", action="store_true", help="Enable detailed local logging")
    return command


def _print_event(event: AgentEvent) -> None:
    if event.type == "status":
        message = event.data.get("message")
        if message:
            print(message, flush=True)
    elif event.type == "tool_started":
        call = event.data.get("call", {})
        print(f"Using {call.get('name', 'tool')}", flush=True)
    elif event.type == "verification_result":
        print("Verified change" if event.data.get("changed") else "No change detected", flush=True)
    elif event.type == "agent_error":
        print(f"Agent error: {event.data.get('message', 'Unknown error')}", flush=True)


async def _approve(request: PermissionRequest) -> bool:
    prompt = f"Allow {request.call.name} ({request.risk.value}: {request.reason})? [y/N] "
    try:
        answer = await asyncio.to_thread(input, prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}


async def _run_cli(args: argparse.Namespace) -> int:
    service = FisherApplication(load_settings())
    options: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "permission_mode": args.permission_mode,
        "profile": args.profile,
        "headless": args.headless,
        "executable_path": args.browser_executable,
    }
    try:
        result = await service.run_once(
            args.task,
            args.url,
            on_event=_print_event,
            approve=_approve,
            **options,
        )
    except KeyboardInterrupt:
        print("Stopped")
        return 130
    except Exception as exc:
        message = str(exc) if isinstance(exc, ProviderError) else f"Task failed ({type(exc).__name__})"
        print(f"Could not run task: {message}")
        return 1
    if result.answer:
        print(result.answer)
    if result.error:
        print(result.error)
    return 0 if result.status == "completed" else 1


def _show_replay(session_id: str) -> int:
    settings = load_settings()
    recorder = SessionRecorder(root=Path(settings.data_dir) / "sessions")
    try:
        session = recorder.get_session(session_id)
    except (FileNotFoundError, ValueError):
        print("Session not found")
        return 1
    print(f"Task: {session.get('task', '')}")
    print(f"Status: {session.get('status', '')}")
    for event in session.get("events", []):
        kind = event.get("type", "")
        data = event.get("data", {})
        if kind == "status":
            print(data.get("message", ""))
        elif kind in {"tool_started", "tool_finished", "agent_finished", "agent_error"}:
            print(kind.replace("_", " "))
    return 0


def _serve(port: int, *, desktop: bool) -> int:
    if not 1 <= port <= 65535:
        print("Port must be between 1 and 65535")
        return 2
    web_index = Path(__file__).resolve().parent / "web" / "index.html"
    if not web_index.is_file():
        print("Frontend is not built. Run npm run build first.")
        return 1
    url = f"http://127.0.0.1:{port}"
    app = create_api()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        if not thread.is_alive():
            print("Could not start the local GUI server")
            return 1
        time.sleep(0.1)
    if not server.started:
        print("Timed out starting the local GUI server")
        return 1
    print(f"Project Fisher control center: {url}")

    try:
        if not desktop:
            thread.join()
            return 0
        if os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            import webview

            webview.create_window("Project Fisher", url=url, width=1400, height=900)
            webview.start()
        else:
            print(f"No desktop display detected. Open {url} in a local browser.")
            thread.join()
    except ImportError:
        print(f"PyWebview is unavailable. Open {url} in a local browser.")
        thread.join()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    return 0


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.replay:
        return _show_replay(args.replay)
    if args.serve or args.gui:
        return _serve(args.port, desktop=args.gui)
    if not args.task:
        parser().error("a task is required (or use --gui, --serve, or --replay)")
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
