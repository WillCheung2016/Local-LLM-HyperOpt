"""Utility helpers for interacting with a local Ollama server."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


class OllamaChatClient:
    """Minimal client for the Ollama chat completion API.

    Parameters
    ----------
    model:
        Name of the Ollama model to use, e.g. ``"llama3"``.
    base_url:
        Base URL where the Ollama server is hosted. Defaults to the value of
        the ``OLLAMA_HOST`` environment variable or ``"http://localhost:11434"``.
    timeout:
        Timeout (in seconds) for HTTP requests to the Ollama server.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, timeout: int = 120) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def _build_options(
        self,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if num_predict is not None:
            options["num_predict"] = num_predict
        if frequency_penalty is not None:
            # Ollama refers to this parameter as ``repeat_penalty``.
            options["repeat_penalty"] = frequency_penalty
        if seed is not None:
            options["seed"] = seed
        return options

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Call the Ollama chat API with the given conversation history.

        Returns the assistant message content from the response. Raises
        ``requests.HTTPError`` if the Ollama server responds with an error or
        ``ValueError`` if the response payload is malformed.
        """

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        options = self._build_options(
            temperature=temperature,
            num_predict=num_predict,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )
        if options:
            payload["options"] = options

        response = requests.post(
            self.base_url.rstrip("/") + "/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        message = data.get("message")
        if not message or "content" not in message:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return message["content"]
