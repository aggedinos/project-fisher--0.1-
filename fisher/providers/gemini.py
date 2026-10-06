"""Gemini REST adapter using header authentication and JSON output mode."""

from __future__ import annotations

import base64
from urllib.parse import quote

from .base import Provider, ProviderError, ProviderOutputError, image_media_type


class GeminiProvider(Provider):
    label = "Gemini"

    def __init__(self, model: str, api_key: str, **kwargs: object) -> None:
        super().__init__(model, **kwargs)
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is required for the Gemini provider")
        self._api_key = api_key

    async def _complete(self, system: str, user: str, image: bytes | None) -> str:
        parts: list[dict[str, object]] = [{"text": user}]
        if image is not None:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": image_media_type(image),
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                }
            )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": 2048,
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(self.model, safe='')}:generateContent"
        result = await self._post_json(url, payload, {"x-goog-api-key": self._api_key})
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderOutputError("Gemini returned no completion candidate")
        first = candidates[0]
        if not isinstance(first, dict):
            raise ProviderOutputError("Gemini returned an invalid candidate")
        content = first.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise ProviderOutputError("Gemini returned no text content")
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and not part.get("thought")
        )
        if not text.strip():
            raise ProviderOutputError("Gemini returned an empty completion")
        return text
