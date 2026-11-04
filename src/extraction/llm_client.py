"""Unified language model client supporting local and cloud backends with failover.

Provides a uniform interface for local llama-server instances (Qwen2.5-3B) and Google Gemini,
handling structured JSON schemas and automatic fallback to local inference on network, timeout,
or quota failures.

Ref: Docs/implementation_plan.md, section 7, WP-08.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from src.core.types import LLM_PROVIDERS
from src.extraction.llm_client_base import (
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMQuotaError,
    build_object_schema,
    coerce_line_id,
    parse_json_object,
)
from src.extraction.llm_providers import (
    GeminiClient,
    OpenAICompatibleClient,
    _sanitize_gemini_schema,
    sanitize_gemini_schema,
)

if TYPE_CHECKING:
    import numpy as np

    from src.core.settings import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "FailoverLLMClient",
    "GeminiClient",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMQuotaError",
    "OpenAICompatibleClient",
    "_sanitize_gemini_schema",
    "build_client",
    "build_object_schema",
    "coerce_line_id",
    "main",
    "parse_json_object",
    "sanitize_gemini_schema",
]


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
        hint_cmd = "run serve" if sys.platform == "win32" else "./run serve"
        print(f"\nConnection Error: {exc}", file=sys.stderr)
        print(f"Hint: Start the local model using: {hint_cmd}", file=sys.stderr)
    except LLMError as exc:
        print(f"\nLLM Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
