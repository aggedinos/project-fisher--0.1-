"""Validated application configuration.

Environment variables override an optional project ``.env`` file. Reading the
configuration never creates an ``.env`` file and never writes secret values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from fisher.models import PermissionMode

from .paths import default_data_dir, profile_dir, sessions_dir


_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2-vision",
    "nvidia": "meta/llama-3.2-90b-vision-instruct",
}


def _boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean setting: {value!r}")


class Settings(BaseModel):
    """Configuration shared by CLI, GUI, browser, and providers."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    provider: Literal["gemini", "ollama", "nvidia"] = "gemini"
    model: str = ""
    gemini_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    nvidia_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    ollama_base_url: str = "http://127.0.0.1:11434"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    permission_mode: PermissionMode = PermissionMode.SAFE
    headless: bool = True
    profile: Literal["temporary", "persistent"] = "temporary"
    executable_path: Path | None = None
    max_steps: int = Field(default=30, ge=1, le=500)
    max_retries: int = Field(default=2, ge=0, le=20)
    allow_private_network: bool = False
    data_dir: Path = Field(default_factory=default_data_dir)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    send_images: bool = True
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=1800.0)

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _default_model(self) -> Settings:
        if not self.model:
            object.__setattr__(self, "model", _DEFAULT_MODELS[self.provider])
        return self

    @property
    def browser_profile_dir(self) -> Path:
        return profile_dir(self.data_dir)

    @property
    def sessions_dir(self) -> Path:
        return sessions_dir(self.data_dir)

    def api_key_for_provider(self) -> str:
        if self.provider == "gemini":
            return self.gemini_api_key.get_secret_value()
        if self.provider == "nvidia":
            return self.nvidia_api_key.get_secret_value()
        return ""

    def with_overrides(self, **overrides: object) -> Settings:
        """Return a fully revalidated copy for CLI or GUI overrides."""
        values = self.model_dump()
        if "provider" in overrides and "model" not in overrides and overrides["provider"] != self.provider:
            values["model"] = ""
        values.update(overrides)
        return type(self).model_validate(values)

    def public_dict(self) -> dict[str, object]:
        """Settings safe to send to a UI or include in a session record."""
        return self.model_dump(mode="json", exclude={"gemini_api_key", "nvidia_api_key"})

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        """Load known settings from `.env` and the process environment.

        A nonexistent file is ordinary; it is never created. Process variables
        take precedence even when explicitly set to an empty string.
        """
        if env_file is None:
            env_file = Path(__file__).resolve().parents[2] / ".env"
        values = {k: v for k, v in dotenv_values(env_file).items() if v is not None} if env_file.is_file() else {}
        values.update(os.environ)
        fields = {
            "provider": "FISHER_PROVIDER",
            "model": "FISHER_MODEL",
            "gemini_api_key": "GEMINI_API_KEY",
            "nvidia_api_key": "NVIDIA_API_KEY",
            "ollama_base_url": "FISHER_OLLAMA_BASE_URL",
            "nvidia_base_url": "FISHER_NVIDIA_BASE_URL",
            "permission_mode": "FISHER_PERMISSION_MODE",
            "headless": "FISHER_HEADLESS",
            "profile": "FISHER_PROFILE",
            "executable_path": "FISHER_EXECUTABLE_PATH",
            "max_steps": "FISHER_MAX_STEPS",
            "max_retries": "FISHER_MAX_RETRIES",
            "allow_private_network": "FISHER_ALLOW_PRIVATE_NETWORK",
            "data_dir": "FISHER_DATA_DIR",
            "temperature": "FISHER_TEMPERATURE",
            "send_images": "FISHER_SEND_IMAGES",
            "timeout_seconds": "FISHER_TIMEOUT_SECONDS",
        }
        kwargs: dict[str, object] = {}
        for field, env_name in fields.items():
            if env_name in values:
                kwargs[field] = values[env_name]
        if "FISHER_PROFILE_MODE" in values:
            kwargs["profile"] = values["FISHER_PROFILE_MODE"]
        for name in ("headless", "allow_private_network", "send_images"):
            if name in kwargs:
                kwargs[name] = _boolean(str(kwargs[name]))
        if kwargs.get("executable_path") == "":
            kwargs["executable_path"] = None
        return cls.model_validate(kwargs)


def load_settings(env_file: Path | None = None) -> Settings:
    return Settings.from_env(env_file)
