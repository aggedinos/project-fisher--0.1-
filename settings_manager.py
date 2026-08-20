"""Local-only JSON-backed settings for machine-specific state.

This module manages hardware / execution context that varies per machine:

  - headless       — run the browser window hidden
  - use_real_chrome — use the persistent Chrome profile
  - canvas_quality  — JPEG quality for the screencast stream (1-100)

Stored at ``paths.settings_file()`` so the desktop app remembers these
across restarts even when installed under a read-only directory.
"""

from __future__ import annotations

import json
from typing import Any

from paths import settings_file

LOCAL_DEFAULTS: dict[str, Any] = {
    "headless": True,
    "use_real_chrome": False,
    "canvas_quality": 60,
}


def load_settings() -> dict[str, Any]:
    """Read ``settings.json``, merging saved values onto the defaults."""
    settings = dict(LOCAL_DEFAULTS)
    path = settings_file()
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = None
        if isinstance(saved, dict):
            for k in LOCAL_DEFAULTS:
                if k in saved:
                    settings[k] = saved[k]
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Write the full local settings dict to disk immediately."""
    settings_file().write_text(json.dumps(settings, indent=2), encoding="utf-8")


def update_setting(key: str, value: Any) -> dict[str, Any]:
    """Persist a single local setting and return the resulting dict."""
    if key not in LOCAL_DEFAULTS:
        raise KeyError(f"{key!r} is not a recognized local setting")
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
    return settings
