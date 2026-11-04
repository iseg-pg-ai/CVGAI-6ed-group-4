"""Shared base interfaces, exceptions, and schema helpers for LLM clients.

Ref: Docs/implementation_plan.md, section 7, WP-08.
"""

from __future__ import annotations

import json
from typing import Any, Protocol


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
