"""Utility helpers for interacting with a local Ollama installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional


class OllamaChatClient:
    """Minimal client for the Ollama chat completion API using the CLI.

    Unlike the previous HTTP-based implementation, this version shells out to
    the ``ollama`` command line interface. This keeps the workflow completely
    headless and avoids the need to expose or access an HTTP endpoint.

    Parameters
    ----------
    model:
        Name of the Ollama model to use, e.g. ``"llama3"``.
    base_url:
        Deprecated parameter retained for backwards compatibility. The value is
        ignored because this client no longer communicates over HTTP.
    executable:
        Optional path to the ``ollama`` executable. If not provided, the value
        of the ``OLLAMA_EXECUTABLE`` environment variable is used, falling back
        to simply ``"ollama"``.
    timeout:
        Timeout (in seconds) for calls to the Ollama CLI.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        *,
        executable: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        if base_url is not None:
            # For backwards compatibility we silently ignore the value but keep
            # it around so existing callers do not break.
            self._deprecated_base_url = base_url
        self.executable = (
            executable
            or os.environ.get("OLLAMA_EXECUTABLE")
            or "ollama"
        )
        if shutil.which(self.executable) is None:
            raise FileNotFoundError(
                "Could not find the 'ollama' executable. Set the OLLAMA_EXECUTABLE "
                "environment variable or pass the executable path explicitly."
            )
        self.timeout = timeout

    @staticmethod
    def _build_options(
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        options: List[str] = []
        if temperature is not None:
            options.extend(["-o", f"temperature={temperature}"])
        if num_predict is not None:
            options.extend(["-o", f"num_predict={num_predict}"])
        if frequency_penalty is not None:
            # Ollama refers to this parameter as ``repeat_penalty``.
            options.extend(["-o", f"repeat_penalty={frequency_penalty}"])
        if seed is not None:
            options.extend(["-o", f"seed={seed}"])
        return options

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt for the CLI."""

        formatted_segments: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue
            role_heading = role.capitalize()
            formatted_segments.append(f"{role_heading}: {content}")
        # Encourage the model to respond as the assistant.
        formatted_segments.append("Assistant:")
        return "\n".join(formatted_segments)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Call the Ollama CLI with the given conversation history."""

        prompt = self._format_messages(messages)
        if not prompt.strip():
            raise ValueError("No content provided in messages for Ollama prompt.")

        cmd = [self.executable, "run", self.model]
        cmd.extend(self._build_options(
            temperature=temperature,
            num_predict=num_predict,
            frequency_penalty=frequency_penalty,
            seed=seed,
        ))
        cmd.append(prompt)

        try:
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Ollama command failed with exit code {exc.returncode}: {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Timed out waiting for Ollama CLI response") from exc

        response_text = process.stdout.strip()
        if not response_text:
            raise ValueError("Received empty response from Ollama CLI")

        # ``ollama run`` may echo the prompt or include JSON fragments when
        # templates are used. We try to return a clean assistant message.
        cleaned = response_text.splitlines()
        # Some models echo the final "Assistant:" prefix. Remove it if present.
        if cleaned and cleaned[0].startswith("Assistant:"):
            cleaned[0] = cleaned[0].replace("Assistant:", "", 1).strip()
        result = "\n".join(line for line in cleaned if line).strip()
        if not result:
            raise ValueError(
                f"Unexpected Ollama response format: {json.dumps(response_text)}"
            )
        return result
