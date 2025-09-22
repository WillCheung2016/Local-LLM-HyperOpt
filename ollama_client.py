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
    def _extract_system_prompt(messages: List[Dict[str, str]]) -> Optional[str]:
        """Return the content of the first system message, if any."""

        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if content:
                    return content
        return None

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt for the CLI."""

        formatted_segments: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                # The system prompt is passed separately to the CLI.
                continue
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

        cmd = [self.executable, "run", self.model, "--json"]
        system_prompt = self._extract_system_prompt(messages)
        if system_prompt:
            cmd.extend(["--system", system_prompt])
        cmd.extend(self._build_options(
            temperature=temperature,
            num_predict=num_predict,
            frequency_penalty=frequency_penalty,
            seed=seed,
        ))

        try:
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                input=prompt,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Ollama command failed with exit code {exc.returncode}: {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Timed out waiting for Ollama CLI response") from exc

        stdout = process.stdout.strip()
        if not stdout:
            raise ValueError("Received empty response from Ollama CLI")

        response_chunks: List[str] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                # Ignore non-JSON lines to be resilient to unexpected output.
                continue

            if payload.get("error"):
                raise RuntimeError(f"Ollama reported an error: {payload['error']}")

            if "response" in payload:
                response_chunks.append(payload["response"])

            if payload.get("done"):
                # Some models include the final assistant message under
                # ``message`` when ``--json`` is used; capture it if present.
                message = payload.get("message")
                if isinstance(message, dict) and message.get("content"):
                    response_chunks.append(message["content"])
                break

        result = "".join(response_chunks).strip()
        if not result:
            raise ValueError(
                f"Unexpected Ollama response format: {json.dumps(stdout)}"
            )
        return result
