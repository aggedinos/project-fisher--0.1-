"""Run the real CLI against local browser and Ollama protocol fixtures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


ROOT = Path(__file__).resolve().parents[2]
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


class _Site(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


class _Ollama(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        system = payload["messages"][0]["content"]
        self.server.requests += 1  # type: ignore[attr-defined]
        if "Make a short plan" in system:
            content = json.dumps({"steps": ["Read the product page", "Report its price"]})
        elif self.server.requests == 2:  # type: ignore[attr-defined]
            content = json.dumps({"tool_call": {"name": "read_page", "arguments": {}}})
        else:
            content = json.dumps({"answer": "The Studio headphones cost $79.95."})
        body = json.dumps({"message": {"content": content}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CliFlowTests(unittest.TestCase):
    def test_cli_uses_ollama_browser_and_tool_registry(self) -> None:
        site = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Site, directory=str(FIXTURES)))
        model = ThreadingHTTPServer(("127.0.0.1", 0), _Ollama)
        model.requests = 0  # type: ignore[attr-defined]
        threads = [
            Thread(target=server.serve_forever, daemon=True) for server in (site, model)
        ]
        for thread in threads:
            thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="fisher-cli-test-") as data_dir:
                env = os.environ.copy()
                env.update({
                    "FISHER_PROVIDER": "ollama",
                    "FISHER_OLLAMA_BASE_URL": f"http://127.0.0.1:{model.server_port}",
                    "FISHER_DATA_DIR": data_dir,
                    "FISHER_ALLOW_PRIVATE_NETWORK": "true",
                    "FISHER_HEADLESS": "true",
                    "PYTHONDONTWRITEBYTECODE": "1",
                })
                if BROWSER_EXECUTABLE:
                    env["FISHER_EXECUTABLE_PATH"] = BROWSER_EXECUTABLE
                run = subprocess.run(
                    [
                        sys.executable, "main.py", "Read the Studio headphones price",
                        "--provider", "ollama",
                        "--url", f"http://127.0.0.1:{site.server_port}/product.html",
                    ],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
                    check=False,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn("$79.95", run.stdout)
                self.assertGreaterEqual(model.requests, 3)  # type: ignore[attr-defined]
                sessions = list((Path(data_dir) / "sessions").glob("*/session.json"))
                self.assertEqual(len(sessions), 1)
                self.assertEqual(json.loads(sessions[0].read_text())["status"], "completed")
        finally:
            for server in (site, model):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
