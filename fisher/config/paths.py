"""Paths for data produced while Fisher runs.

Source checkouts and installed application files are never used for sessions,
screenshots, or browser profiles. Merely importing this module creates nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def default_data_dir() -> Path:
    """Return a suitable per-user data directory for the current platform."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "Project Fisher"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Project Fisher"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "project-fisher"


def ensure_directory(path: Path) -> Path:
    """Create a private runtime directory when the caller actually needs it."""
    path = path.expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def profile_dir(data_dir: Path) -> Path:
    """Fisher-owned persistent browser profile (never a user's Chrome profile)."""
    return data_dir / "browser-profile"


def sessions_dir(data_dir: Path) -> Path:
    return data_dir / "sessions"
