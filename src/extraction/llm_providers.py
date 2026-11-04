"""Concrete backend clients for OpenAI-compatible endpoints and Google Gemini.

Ref: Docs/implementation_plan.md, section 7, WP-08.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.extraction.llm_client_base import LLMConnectionError, LLMError, LLMQuotaError

if TYPE_CHECKING:
    import numpy as np


class OpenAICompatibleClient:
    """Client for backends adhering to the OpenAI /v1/chat/completions specification.

    Serves local llama-server instances.
    """

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        client: Any | None = None,
    ) -> None:
        """Initialise the OpenAI-compatible client.

        Args:
            provider: Identifier of the provider, e.g. 'llm_local'.
            base_url: Target API endpoint URL.
            api_key: API authorization key.
            model: Model name string.
            timeout_s: Request timeout in seconds.
            client: Optional pre-configured client instance for testing or dependency injection.
        """
        self.provider = provider
        self.last_effective_provider = provider
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s

        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        image: np.ndarray | None = None,
    ) -> str:
        """Generate completion using OpenAI chat completions API."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_type = type(exc).__name__
            err_str = str(exc)
            if "Connection" in err_type or "ConnectError" in err_type or "Connection refused" in err_str:
                msg = (
                    f"Could not connect to {self.provider} at {self.base_url} (Connection refused). "
                    "Ensure llama-server is running (see scripts/run_llama_server.sh)."
                    if self.provider == "llm_local"
                    else f"Could not connect to {self.provider} at {self.base_url}."
                )
                raise LLMConnectionError(msg) from exc
            if "Timeout" in err_type or "timeout" in err_str.lower():
                raise LLMConnectionError(f"Request to {self.provider} at {self.base_url} timed out.") from exc
            if "RateLimit" in err_type or "429" in err_str:
                raise LLMQuotaError(f"Rate limit exceeded for {self.provider} ({exc}).") from exc
            raise LLMError(f"API error from {self.provider}: {exc}") from exc

        self.last_effective_provider = self.provider

        content = response.choices[0].message.content
        return content or ""


def sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively strip fields unsupported by Google Gemini's Schema API.

    OpenAI strict mode mandates 'additionalProperties: false', but Google Gemini's
    Schema API does not support 'additionalProperties' and rejects requests with
    HTTP 400 INVALID_ARGUMENT.
    """
    if isinstance(schema, dict):
        return {
            k: sanitize_gemini_schema(v)
            for k, v in schema.items()
            if k not in ("additionalProperties", "additional_properties")
        }
    if isinstance(schema, list):
        return [sanitize_gemini_schema(item) for item in schema]
    return schema


# Alias for backward compatibility
_sanitize_gemini_schema = sanitize_gemini_schema


class GeminiClient:
    """Client for Google Gemini models using the google-genai SDK."""

    provider: str = "llm_gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        client: Any | None = None,
    ) -> None:
        """Initialise the Google Gemini client.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model identifier.
            client: Optional pre-configured client instance for testing or dependency injection.
        """
        self.last_effective_provider = self.provider
        self.model = model

        if client is not None:
            self._client = client
        else:
            from google import genai

            self._client = genai.Client(api_key=api_key)

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        image: np.ndarray | None = None,
    ) -> str:
        """Generate completion using Google GenAI SDK."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": 0.0,
        }

        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = sanitize_gemini_schema(schema)

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=config,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                raise LLMQuotaError(f"Quota exceeded for {self.provider} ({exc}).") from exc
            if "connect" in err_str or "timeout" in err_str:
                raise LLMConnectionError(f"Connection to {self.provider} failed ({exc}).") from exc
            raise LLMError(f"Gemini error: {exc}") from exc

        self.last_effective_provider = self.provider
        return response.text or ""
