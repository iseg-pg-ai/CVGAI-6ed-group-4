"""Unified language model client supporting local and cloud backends with failover.

Provides a uniform interface for local llama-server instances (Qwen2.5-3B) and Google Gemini,
handling structured JSON schemas and automatic fallback to local inference on network, timeout,
or quota failures.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-08.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from src.core.types import LLM_PROVIDERS

if TYPE_CHECKING:
    import numpy as np

    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base exception for all language model client failures."""


class LLMConnectionError(LLMError):
    """Raised when an LLM provider endpoint is unreachable (connection refused, timeout)."""


class LLMQuotaError(LLMError):
    """Raised when an API quota is exhausted (HTTP 429)."""


def build_object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Build a closed JSON object schema with every property required.

    ``additionalProperties: false`` ensures strict schema enforcement on local
    llama-server grammar constraints and is automatically stripped by GeminiClient.

    Args:
        properties: Property schemas keyed by property name.

    Returns:
        A JSON schema object requiring all given properties.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def parse_json_object(raw: str, error_cls: type[Exception] = ValueError) -> dict[str, Any]:
    """Recover a JSON object from a raw model response string.

    Tolerates markdown code fences and prose surrounding the JSON object.

    Args:
        raw: Raw text returned by the LLM client.
        error_cls: Exception class to raise on malformed JSON or non-dict payloads.

    Returns:
        Decoded JSON dictionary.

    Raises:
        Exception: Instance of error_cls when no JSON object can be decoded.
    """
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise error_cls(f"Response contains no JSON object: {text[:200]!r}")

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise error_cls(f"Response JSON is a {type(payload).__name__}, expected an object")
    return payload


def coerce_line_id(raw: object, valid_ids: set[int]) -> int | None:
    """Coerce a raw line identifier to an int if it exists in valid_ids.

    Args:
        raw: Candidate line identifier (int or numeric string) from decoded JSON.
        valid_ids: Set of valid 1-based line identifiers in the OCR document.

    Returns:
        Validated integer line ID, or None if invalid or absent from valid_ids.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw in valid_ids else None
    if isinstance(raw, str) and raw.strip().isdigit():
        candidate = int(raw.strip())
        return candidate if candidate in valid_ids else None
    return None


class LLMClient(Protocol):
    """Interface every LLM provider backend must satisfy."""

    provider: str
    last_effective_provider: str

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion from system and user prompt strings.

        Args:
            system: System instructions (persona, task definition, schema description).
            user: User content (indexed OCR lines, questions, document context).
            schema: Optional flat JSON schema dictionary for structured output.

        Returns:
            Raw text response from the model (structured JSON string or natural language).
        """
        ...


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


def _sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively strip fields unsupported by Google Gemini's Schema API.

    OpenAI strict mode mandates 'additionalProperties: false', but Google Gemini's
    Schema API does not support 'additionalProperties' and rejects requests with
    HTTP 400 INVALID_ARGUMENT.
    """
    if isinstance(schema, dict):
        return {
            k: _sanitize_gemini_schema(v)
            for k, v in schema.items()
            if k not in ("additionalProperties", "additional_properties")
        }
    if isinstance(schema, list):
        return [_sanitize_gemini_schema(item) for item in schema]
    return schema


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
            config_kwargs["response_schema"] = _sanitize_gemini_schema(schema)

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


class FailoverLLMClient:
    """Wrapper that catches cloud provider errors and transparently falls back to local LLM.

    Guarantees that live classroom demonstrations or offline runs never crash due to network
    outages, DNS failures, or remote quota exhaustion.
    """

    def __init__(
        self,
        provider: str,
        primary: LLMClient | None,
        fallback: LLMClient,
        has_api_key: bool = True,
    ) -> None:
        """Initialise the failover wrapper.

        Args:
            provider: Advertised provider name (e.g. 'llm_gemini').
            primary: Primary cloud client instance or None when unconfigured.
            fallback: Robust local client instance (llama-server).
            has_api_key: Whether a valid API key was provided for the primary client.
        """
        self.provider = provider
        self.primary = primary
        self.fallback = fallback
        self.has_api_key = has_api_key
        self.last_effective_provider = provider if has_api_key else fallback.provider

    def _call_fallback(self, system: str, user: str, kwargs: dict[str, Any], failure_prefix: str) -> str:
        """Execute completion on the fallback client and update effective provider."""
        try:
            result = self.fallback.complete(system, user, **kwargs)
            self.last_effective_provider = self.fallback.last_effective_provider
            return result
        except LLMConnectionError as fb_exc:
            raise LLMConnectionError(f"{failure_prefix}: {fb_exc}") from fb_exc

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        image: np.ndarray | None = None,
    ) -> str:
        """Execute request with primary provider or fallback to local on any failure."""
        kwargs: dict[str, Any] = {"schema": schema}
        if image is not None:
            kwargs["image"] = image

        if not self.has_api_key or self.primary is None:
            logger.warning(
                "No API key configured for %s; falling back to %s.",
                self.provider,
                self.fallback.provider,
            )
            return self._call_fallback(
                system,
                user,
                kwargs,
                f"No API key configured for {self.provider}, and fallback to {self.fallback.provider} failed",
            )

        try:
            result = self.primary.complete(system, user, **kwargs)
            self.last_effective_provider = self.primary.last_effective_provider
            return result
        except Exception as exc:
            logger.warning(
                "Primary provider %s failed with %s; falling back to %s.",
                self.provider,
                exc,
                self.fallback.provider,
            )
            return self._call_fallback(
                system,
                user,
                kwargs,
                f"Primary provider {self.provider} failed ({exc}), and fallback to {self.fallback.provider} also failed",
            )


def build_client(
    provider: str = "llm_local",
    settings: Settings | None = None,
    primary_client: Any | None = None,
    fallback_client: Any | None = None,
) -> LLMClient:
    """Factory creating configured LLM clients with failover safety.

    Args:
        provider: Provider identifier ('llm_local' or 'llm_gemini').
        settings: Optional Settings instance. When None, get_settings() is used.
        primary_client: Optional injected primary client for unit testing.
        fallback_client: Optional injected fallback client for unit testing.

    Returns:
        Configured LLMClient instance satisfying the protocol.
    """
    from src.core.settings import get_settings

    cfg = settings or get_settings()

    local_client = fallback_client or OpenAICompatibleClient(
        provider="llm_local",
        base_url=cfg.llama_base_url,
        api_key="sk-no-key-required",
        model=cfg.llama_model,
        timeout_s=cfg.llama_timeout_s,
    )

    match provider:
        case "llm_local":
            return local_client

        case "llm_gemini":
            has_key = bool(cfg.gemini_api_key) or primary_client is not None
            gemini_client = primary_client or (
                GeminiClient(
                    api_key=cfg.gemini_api_key or "dummy-unconfigured",
                    model=cfg.gemini_model,
                )
                if has_key
                else None
            )
            return FailoverLLMClient(
                provider="llm_gemini",
                primary=gemini_client,
                fallback=local_client,
                has_api_key=has_key,
            )

        case _:
            # Unknown provider fallback
            logger.warning("Unknown provider '%s'; defaulting to local LLM client.", provider)
            return local_client


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to test LLM client connectivity and responses.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    parser = argparse.ArgumentParser(description="Unified LLM client preview")
    parser.add_argument(
        "--provider",
        "-p",
        choices=LLM_PROVIDERS,
        default="llm_local",
        help="LLM provider backend to query",
    )
    parser.add_argument(
        "--prompt",
        default="Responde apenas com a palavra 'OK' para confirmar funcionamento.",
        help="User prompt text",
    )
    parser.add_argument(
        "--system",
        "-s",
        default="Es um assistente util e conciso.",
        help="System instruction text",
    )

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    print(f"Connecting to provider: {args.provider}...")
    client = build_client(provider=args.provider, settings=settings_override)

    try:
        start_time = time.perf_counter()
        response = client.complete(system=args.system, user=args.prompt)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"Response received in {elapsed_ms:.1f} ms:")
        print(f"  Effective provider: {client.last_effective_provider}")
        print(f"  Output: {response}")
    except LLMConnectionError as exc:
        hint_cmd = r".\scripts\run_llama_server.ps1" if sys.platform == "win32" else "./scripts/run_llama_server.sh"
        print(f"\nConnection Error: {exc}", file=sys.stderr)
        print(f"Hint: Start the local model using: {hint_cmd}", file=sys.stderr)
    except LLMError as exc:
        print(f"\nLLM Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
