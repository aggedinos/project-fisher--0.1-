"""NVIDIA OpenAI-compatible chat adapter."""

from __future__ import annotations

import base64

from .base import Provider, ProviderError, ProviderOutputError, image_media_type


class NvidiaProvider(Provider):
    label = "NVIDIA"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        **kwargs: object,
    ) -> None:
        super().__init__(model, **kwargs)
        if not api_key:
            raise ProviderError("NVIDIA_API_KEY is required for the NVIDIA provider")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def _complete(self, system: str, user: str, image: bytes | None) -> str:
        content: str | list[dict[str, object]] = user
        if image is not None:
            encoded = base64.b64encode(image).decode("ascii")
            media_type = image_media_type(image)
            content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
            ]
        result = await self._post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "temperature": self.temperature,
                "max_tokens": 2048,
                "stream": False,
            },
            {"Authorization": f"Bearer {self._api_key}"},
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderOutputError("NVIDIA returned no completion choice")
        message = choices[0].get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if isinstance(text, list):
            text = "".join(
                part["text"]
                for part in text
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(text, str) or not text.strip():
            raise ProviderOutputError("NVIDIA returned an empty completion")
        return text
