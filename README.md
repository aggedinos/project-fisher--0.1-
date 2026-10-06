# Project Fisher 0.2

Project Fisher is a local browser agent with no Fisher account or sign-in page. Give it a task and watch it inspect pages and operate Playwright browser tools. Its React control center shows the plan, current action, browser preview, permission requests, and recorded sessions. The same agent runs from the CLI. Ollama is the default provider; Gemini and NVIDIA remain optional.

Fisher treats page text as untrusted data. A model selects a named, typed tool; Fisher validates its arguments and permissions, executes it, and checks whether the page changed. It stops after bounded retries and reports blocked or partial work instead of looping indefinitely.

## Requirements

- Python 3.11 or newer
- Node.js 20.19 or newer on the 20.x line, or 22.12 or newer, for building the UI
- Chromium installed through Playwright, or a compatible local Chromium executable
- A running local Ollama server and model for the default account-free setup; Gemini or NVIDIA API keys only if you choose those optional providers
- A graphical desktop for the native PyWebview window. The local web control center also works in a browser.

## Install

From the repository root on Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
npm ci
npm run build
.\run_gui.bat
```

On Linux or macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
npm ci
npm run build
```

If Playwright's Chromium download is unavailable but Chromium is installed locally, set `FISHER_EXECUTABLE_PATH` or pass `--browser-executable` to the CLI. Use a Chromium version compatible with the installed Playwright release.

Settings are saved only on this device in the per-user data directory (`settings.json`); no Fisher account or remote settings service is involved. Ollama requires no API key. To use an optional cloud provider, copy [`.env.example`](.env.example) to an untracked `.env` and set its key locally. Process environment variables take precedence over `.env` and saved control center settings. Fisher never creates or writes a credentials file.

```dotenv
GEMINI_API_KEY=your-local-key
# Or use NVIDIA_API_KEY=...
# Ollama remains the default when no key is set
```

The example above is a template; keep real keys out of version control. The runtime keeps sessions, settings, and its own persistent browser profile in a per-user data directory. Override it with `FISHER_DATA_DIR` if needed.

## Run the control center

```bash
python main.py --gui
```

On Windows, `run_gui.bat` runs the command from its own directory. On a desktop, `--gui` opens a PyWebview window. Without a desktop display, it serves the same UI at `http://127.0.0.1:8000` for a local browser. `python main.py --serve` serves it without opening a native window. The server binds to loopback only.

Enter a task and optional starting URL, then choose Run. The right panel shows the plan, action, and event history. Stop cancels the task. When a consequential action needs approval, the dialog offers Approve and Decline. Settings select the provider, model, permission mode, browser profile, and headless mode; no API key is sent back to the UI.

## CLI

```bash
python main.py "Summarize this page" --url https://example.com
python main.py "Find the Python documentation" --provider gemini --url https://www.python.org
python main.py "Inspect this page" --provider nvidia --permission-mode supervised --headed
```

Useful options: `--url`, `--provider`, `--model`, `--permission-mode`, `--profile`, `--headless` or `--headed`, `--browser-executable`, `--replay`, `--gui`, and `--serve`. Run `python main.py --help` for the complete list.

CLI and GUI use the same browser controller, tool registry, providers, planner, verifier, and session recorder. In a terminal, permission requests require an explicit `y`; EOF or an empty answer declines.

### Ollama

Install and start [Ollama](https://ollama.com/), then pull a vision capable model such as `llama3.2-vision`:

```bash
ollama pull llama3.2-vision
python main.py "Describe the current page" --provider ollama --model llama3.2-vision
```

The default Ollama endpoint is `http://127.0.0.1:11434`; change it with `FISHER_OLLAMA_BASE_URL`. Fisher does not start or install Ollama.

## Permissions and browser safety

| Mode | Routine actions | Medium risk actions | High risk actions |
| --- | --- | --- | --- |
| `auto` | automatic | automatic | ask |
| `safe` (default) | automatic | ask | ask |
| `supervised` | ask for meaningful actions | ask | ask |

Reading a page never needs approval. High risk actions include consequential form submission, purchases, deletion, publishing, credential entry, and uncertain coordinate clicks. Downloads are disabled in the browser context. CAPTCHA and human verification are left to the user; Fisher does not solve or bypass them.

Fisher accepts HTTP(S) navigation and blocks localhost and private network browser destinations by default. For a trusted local fixture, set `FISHER_ALLOW_PRIVATE_NETWORK=true` explicitly. This setting changes browser navigation policy, not the Ollama connection. Fisher uses its own temporary browser or Fisher-owned persistent profile; it never kills Chrome processes or opens your regular Chrome profile.

Webpage content is labeled untrusted in model prompts and cannot grant permission or change Fisher's policy. Avoid placing secrets in task text or ordinary form fields. Session files redact entered values and known credential patterns. Observed password, payment, and verification fields are covered in browser previews and model screenshots; session screenshots are disabled by default and stored separately when enabled.

## Sessions and replay

Runs are recorded under the user data directory's `sessions/` folder. Each session has a compact `session.json` and may have separate screenshot files. Select Sessions in the control center to replay the event history, or run:

```bash
python main.py --replay session-000001
```

Replay is read only; it never repeats clicks, submissions, or other browser actions.

## Architecture

- `fisher/browser`: Playwright lifecycle, tabs, compact page perception, short-lived semantic element IDs, JPEG preview.
- `fisher/tools` and `fisher/security`: validated tool schemas, URL and risk policy, permission gate, timeouts, structured results.
- `fisher/providers`: one asynchronous interface for Gemini, Ollama, and NVIDIA.
- `fisher/agent`: planning, bounded memory, observe → act → verify loop, recovery, cancellation.
- `fisher/sessions`: sanitized recording and read-only replay.
- `fisher/app.py` and `fisher/api.py`: shared application service and loopback UI contract.
- `frontend/`: React, TypeScript, and Vite source. `npm run build` emits `web/`.

## Development and tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m unittest discover -s tests/unit -p 'test_*.py'
python -m unittest discover -s tests/integration -p 'test_*.py'
ruff check fisher main.py tests
ruff format fisher main.py tests
npm run lint
npm run build
```

The browser integration suite serves deterministic local pages and uses Chromium. It uses `FISHER_TEST_BROWSER`, then `FISHER_EXECUTABLE_PATH`, then a Chromium executable on `PATH`, and finally Playwright's bundled browser. The tests explicitly allow their local fixture addresses.

## Troubleshooting

- **Browser launch fails:** run `python -m playwright install chromium`, or set `FISHER_EXECUTABLE_PATH` to a compatible Chromium executable.
- **Cloud provider unavailable:** configure `GEMINI_API_KEY` or `NVIDIA_API_KEY` in your private environment. For Ollama, verify its local server and model are running.
- **UI missing:** run `npm ci && npm run build` from the repository root.
- **Build cannot find `/src/main.tsx`:** you have the older 0.1 source. Use the 0.2 checkout, which contains `frontend/main.tsx`, before running `npm ci` and `npm run build`.
- **A task is blocked:** check the permission dialog, URL policy, or event log. A failed interaction is re-observed and retried only within its configured budget.
- **No desktop display:** use `python main.py --serve` and open the local loopback URL in a browser.

## License

Project Fisher is covered by the [Project Fisher Proprietary License](LICENSE), not an open source or MIT license. Read the license before using or distributing the software.
