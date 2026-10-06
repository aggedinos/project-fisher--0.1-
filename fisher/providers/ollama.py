"""Local Ollama chat adapter."""

from __future__ import annotations

import base64

from .base import Provider, ProviderOutputError


class OllamaProvider(Provider):
    label = "Ollama"

    def __init__(
        self, model: str, base_url: str = "http://127.0.0.1:11434", **kwargs: object
    ) -> None:
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")

    async def _complete(self, system: str, user: str, image: bytes | None) -> str:
        user_message: dict[str, object] = {"role": "user", "content": user}
        if image is not None:
            user_message["images"] = [base64.b64encode(image).decode("ascii")]
        result = await self._post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, user_message],
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature, "num_predict": 2048},
            },
            {},
        )
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderOutputError("Ollama returned an empty completion")
        return content
