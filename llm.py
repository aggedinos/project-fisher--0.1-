"""LLM provider abstraction — a single interface over Gemini and local Ollama.

The agent keeps a provider-agnostic conversation history (a list of
:class:`Message`). Each provider knows how to translate that history into its
own wire format, send it, and return the raw text reply.
"""

from __future__ import annotations

import asyncio
import base64
import os
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """One turn of the conversation, independent of any provider.

    ``role`` is ``"user"`` or ``"model"``. ``image`` holds raw PNG bytes and is
    only ever set on user turns (the screenshot for that step).
    """

    role: str
    text: str
    image: bytes | None = None


class LLMProvider(ABC):
    """Common interface every backend implements."""

    
    label: str = "llm"

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        history: list[Message],
        session_token: str | None = None,
    ) -> str:
        """Send ``system_prompt`` + ``history`` and return the raw text reply."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The active model name."""







class GeminiProvider(LLMProvider):
    """Google Gemini backend (cloud). Calls the Gemini REST API directly."""

    label = "Gemini"

    def __init__(
        self,
        model: str,
        temperature: float,
    ) -> None:
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        history: list[Message],
        session_token: str | None = None,
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )

        
        contents = []
        for msg in history:
            parts: list[dict] = [{"text": msg.text}]
            if msg.image is not None:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(msg.image).decode("ascii"),
                    }
                })
            role = "model" if msg.role == "model" else "user"
            contents.append({"role": role, "parts": parts})

        payload: dict = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={api_key}"
        )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0,
                )
            except httpx.RequestError as exc:
                raise RuntimeError(
                    f"Connection to Gemini API failed: {exc}"
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Gemini API returned error ({response.status_code}): {response.text}"
                )

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No completion candidates returned from Gemini.")

            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)







class OllamaProvider(LLMProvider):
    """Local Ollama backend — talks to the Ollama HTTP API on this machine.

    For the agent to "see" screenshots the chosen model must be vision-capable
    (e.g. ``llama3.2-vision``, ``llava``, ``qwen2.5vl``, ``moondream``).
    Text-only models still work using the extracted page text; set
    ``send_images=False`` so they aren't sent images they can't read.
    """

    label = "Ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        send_images: bool = True,
        timeout: int = 600,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._send_images = send_images
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _to_messages(self, system_prompt: str, history: list[Message]):
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            entry: dict = {"role": role, "content": msg.text}
            if msg.image is not None and self._send_images:
                entry["images"] = [base64.b64encode(msg.image).decode("ascii")]
            messages.append(entry)
        return messages

    def _post_chat(self, system_prompt: str, history: list[Message]) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": self._to_messages(system_prompt, history),
            "stream": False,
            "format": "json",  
            "options": {"temperature": self._temperature},
        }
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "") or ""

    async def generate(
        self,
        system_prompt: str,
        history: list[Message],
        session_token: str | None = None,
    ) -> str:
        return await asyncio.to_thread(self._post_chat, system_prompt, history)

    @staticmethod
    def list_models(base_url: str, timeout: int = 5) -> list[str]:
        """Return the names of models installed in the local Ollama server."""
        import requests

        resp = requests.get(
            f"{base_url.rstrip('/')}/api/tags", timeout=timeout
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return sorted(m["name"] for m in models if m.get("name"))







class NvidiaProvider(LLMProvider):
    """NVIDIA-hosted models via the OpenAI-compatible API."""

    label = "NVIDIA"

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        send_images: bool = True,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._send_images = send_images
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _to_messages(self, system_prompt: str, history: list[Message]):
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = "user" if msg.role == "user" else "assistant"
            if msg.image is not None and self._send_images:
                b64 = base64.b64encode(msg.image).decode("ascii")
                content: object = [
                    {"type": "text", "text": msg.text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            else:
                content = msg.text
            messages.append({"role": role, "content": content})
        return messages

    def _post_chat(self, system_prompt: str, history: list[Message]) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": self._to_messages(system_prompt, history),
            "temperature": self._temperature,
            "max_tokens": 4096,
            "stream": False,
        }
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"NVIDIA API {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _keep_only_last_image(history: list[Message]) -> list[Message]:
        last_img = next(
            (
                i
                for i in range(len(history) - 1, -1, -1)
                if history[i].image is not None
            ),
            None,
        )
        return [
            msg
            if msg.image is None or i == last_img
            else Message(role=msg.role, text=msg.text, image=None)
            for i, msg in enumerate(history)
        ]

    async def generate(
        self,
        system_prompt: str,
        history: list[Message],
        session_token: str | None = None,
    ) -> str:
        history = self._keep_only_last_image(history)
        return await asyncio.to_thread(self._post_chat, system_prompt, history)

    @staticmethod
    def list_models(
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 10,
    ) -> list[str]:
        import requests

        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return sorted(
            m["id"] for m in resp.json().get("data", []) if m.get("id")
        )


def build_provider(
    *,
    provider: str,
    temperature: float,
    gemini_model: str = "gemini-2.5-flash",
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.2-vision",
    nvidia_api_key: str = "",
    nvidia_model: str = "meta/llama-3.2-90b-vision-instruct",
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1",
    send_images: bool = True,
    session_token: str | None = None,
) -> LLMProvider:
    """Factory used by both the CLI and the GUI to construct a provider."""
    provider = provider.lower().strip()
    if provider == "gemini":
        return GeminiProvider(
            model=gemini_model,
            temperature=temperature,
        )
    if provider == "ollama":
        return OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=temperature,
            send_images=send_images,
        )
    if provider == "nvidia":
        if not nvidia_api_key:
            raise ValueError(
                "NVIDIA selected but no API key provided. Set NVIDIA_API_KEY "
                "in .env or paste a key in the GUI."
            )
        return NvidiaProvider(
            api_key=nvidia_api_key,
            model=nvidia_model,
            temperature=temperature,
            base_url=nvidia_base_url,
            send_images=send_images,
        )
    raise ValueError(
        f"Unknown provider: {provider!r} (use 'gemini', 'ollama', or 'nvidia')"
    )
