"""Runtime path resolution for source runs and the frozen installer.

When packaged by PyInstaller and installed under Program Files the install
directory is read-only for standard users, so every writable file (the
``.env`` with API keys, ``debug/`` screenshots, ``sessions/`` recordings)
lives in a per-user data directory instead. Run from source, behaviour is
unchanged — those files stay next to the project as before.

Importing this module also points Playwright at the Chromium that ships
inside the installed app, so a packaged build never tries to download a
browser.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "Browser Automation Agent"



_DEFAULT_ENV = """\
# Browser Automation Agent - API keys (leave blank to use Ollama locally).
#
# NVIDIA (cloud) - get a key at https://build.nvidia.com
NVIDIA_API_KEY=
"""


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Directory holding bundled resources.

    ``sys._MEIPASS`` is set by PyInstaller for both one-file and (since
    PyInstaller 6) one-dir builds; from source this is the project folder.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Per-user writable directory, created on demand.

    Frozen: ``%LOCALAPPDATA%\\Browser Automation Agent``. Source: the
    project directory (unchanged behaviour for developers).
    """
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = Path(base) / APP_DIR_NAME
    else:
        path = _project_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def writable_subdir(name: str) -> str:
    """Return (creating) a writable subdirectory under the data dir."""
    path = user_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def chrome_profile_dir() -> str:
    """Persistent Chrome profile used by ``launch_persistent_context``.

    Lives under the user data dir so cookies, logins, and extensions survive
    across runs without polluting the user's real Chrome profile (Chrome
    holds an exclusive lock on its own ``User Data`` directory).
    """
    return writable_subdir("chrome_profile")


def env_file() -> Path:
    """Path to the user's ``.env``; writes an empty default on first run."""
    path = (user_data_dir() if is_frozen() else _project_dir()) / ".env"
    if not path.exists():
        try:
            path.write_text(_DEFAULT_ENV, encoding="utf-8")
        except OSError:
            pass
    return path


def settings_file() -> Path:
    """Path to the persisted UI/agent settings (theme, engine tier, browser toggles)."""
    return user_data_dir() / "settings.json"


def _setup_playwright_browsers() -> None:
    """Point Playwright at the Chromium bundled inside the installed app.

    No-op from source (Playwright uses its normal per-user cache). When
    frozen, the installer ships ``playwright-browsers/`` next to the app.
    """
    if not is_frozen():
        return
    bundled = bundle_dir() / "playwright-browsers"
    if bundled.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(bundled)
        os.environ.setdefault("PLAYWRIGHT_SKIP_BROWSER_GC", "1")


def init_runtime() -> None:
    """Prepare environment for a packaged run. Safe to call repeatedly."""
    _setup_playwright_browsers()




init_runtime()
