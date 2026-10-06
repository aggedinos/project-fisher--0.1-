"""Construct the chosen cloud or local model provider."""

from __future__ import annotations

import httpx

from fisher.config.settings import Settings

from .base import Provider, ProviderError, ProviderOutputError, ProviderTransportError
from .gemini import GeminiProvider
from .nvidia import NvidiaProvider
from .ollama import OllamaProvider


def build_provider(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> Provider:
    common = {
        "temperature": settings.temperature,
        "timeout_seconds": settings.timeout_seconds,
        "send_images": settings.send_images,
        "transport": transport,
    }
    if settings.provider == "gemini":
        return GeminiProvider(settings.model, settings.api_key_for_provider(), **common)
    if settings.provider == "ollama":
        return OllamaProvider(settings.model, settings.ollama_base_url, **common)
    if settings.provider == "nvidia":
        return NvidiaProvider(
            settings.model, settings.api_key_for_provider(), settings.nvidia_base_url, **common
        )
    raise ProviderError("Unknown model provider")


__all__ = [
    "Provider",
    "ProviderError",
    "ProviderOutputError",
    "ProviderTransportError",
    "GeminiProvider",
    "OllamaProvider",
    "NvidiaProvider",
    "build_provider",
]
