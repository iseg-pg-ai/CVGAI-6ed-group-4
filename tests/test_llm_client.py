"""Unit and contract tests for the unified LLM client and failover wrapper.

All tests run strictly offline, deterministically, and in milliseconds by injecting
mock client backends. No real API keys or running llama-server instances are required.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-08.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.llm_client import (
    FailoverLLMClient,
    GeminiClient,
    LLMClient,
    OpenAICompatibleClient,
    build_client,
    main,
)
from src.settings import Settings


def make_test_settings(
    gemini_key: str | None = None,
) -> Settings:
    """Helper creating an isolated Settings instance for client tests."""
    from pathlib import Path

    return Settings(
        llama_base_url="http://127.0.0.1:8080/v1",
        llama_model="qwen2.5-3b",
        llama_timeout_s=30.0,
        gemini_api_key=gemini_key,
        gemini_model="gemini-2.5-flash",
        ocr_engine="easyocr",
        easyocr_langs=("pt", "en"),
        tesseract_cmd=None,
        repo_root=Path("/fake/root"),
        data_dir=Path("/fake/data"),
        receipts_dir=Path("/fake/receipts"),
        ground_truth_path=Path("/fake/gt.csv"),
        results_dir=Path("/fake/results"),
    )


def test_openai_compatible_client_unstructured() -> None:
    """OpenAICompatibleClient formats system and user prompts and returns message text."""
    mock_inner = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "RESPOSTA LOCAL"
    mock_inner.chat.completions.create.return_value.choices = [mock_choice]

    client = OpenAICompatibleClient(
        provider="llm_local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        model="qwen2.5-3b",
        client=mock_inner,
    )

    out = client.complete(system="Sys prompt", user="User prompt")

    assert out == "RESPOSTA LOCAL"
    assert client.last_effective_provider == "llm_local"

    call_kwargs = mock_inner.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "qwen2.5-3b"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "Sys prompt"},
        {"role": "user", "content": "User prompt"},
    ]
    assert "response_format" not in call_kwargs


def test_openai_compatible_client_with_schema() -> None:
    """OpenAICompatibleClient wraps json_schema in response_format."""
    mock_inner = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"campo": "valor"}'
    mock_inner.chat.completions.create.return_value.choices = [mock_choice]

    client = OpenAICompatibleClient(
        provider="llm_local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        model="qwen2.5-3b",
        client=mock_inner,
    )

    schema = {"type": "object", "properties": {"campo": {"type": "string"}}}
    out = client.complete(system="Sys", user="User", schema=schema)

    assert out == '{"campo": "valor"}'
    call_kwargs = mock_inner.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": schema},
    }


def test_gemini_client_with_and_without_schema() -> None:
    """GeminiClient formats system instructions and schema config correctly."""
    mock_inner = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"total": "16.10"}'
    mock_inner.models.generate_content.return_value = mock_resp

    client = GeminiClient(api_key="gemini-key", model="gemini-2.5-flash", client=mock_inner)

    schema = {"type": "object", "properties": {"total": {"type": "string"}}}
    out = client.complete(system="System instructions", user="User query", schema=schema)

    assert out == '{"total": "16.10"}'
    assert client.provider == "llm_gemini"

    call_kwargs = mock_inner.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "User query"
    config = call_kwargs["config"]
    assert config.system_instruction == "System instructions"
    assert config.response_mime_type == "application/json"
    assert config.response_schema == schema


def test_failover_when_api_key_missing() -> None:
    """FailoverLLMClient delegates directly to fallback when primary API key is missing."""
    mock_fallback = MagicMock(spec=LLMClient)
    mock_fallback.provider = "llm_local"
    mock_fallback.last_effective_provider = "llm_local"
    mock_fallback.complete.return_value = "FALLBACK RESPONSE"

    failover = FailoverLLMClient(
        provider="llm_gemini",
        primary=None,
        fallback=mock_fallback,
        has_api_key=False,
    )

    out = failover.complete(system="Sys", user="User")

    assert out == "FALLBACK RESPONSE"
    assert failover.last_effective_provider == "llm_local"
    mock_fallback.complete.assert_called_once_with("Sys", "User", schema=None)


def test_failover_on_network_or_api_exception() -> None:
    """FailoverLLMClient catches primary client exceptions and falls back to local."""
    mock_primary = MagicMock(spec=LLMClient)
    mock_primary.provider = "llm_gemini"
    mock_primary.complete.side_effect = ConnectionError("Quota exceeded or connection refused")

    mock_fallback = MagicMock(spec=LLMClient)
    mock_fallback.provider = "llm_local"
    mock_fallback.last_effective_provider = "llm_local"
    mock_fallback.complete.return_value = "RECOVERED VIA LOCAL"

    failover = FailoverLLMClient(
        provider="llm_gemini",
        primary=mock_primary,
        fallback=mock_fallback,
        has_api_key=True,
    )

    out = failover.complete(system="Sys", user="User")

    assert out == "RECOVERED VIA LOCAL"
    assert failover.last_effective_provider == "llm_local"
    mock_primary.complete.assert_called_once()
    mock_fallback.complete.assert_called_once()


def test_failover_uses_primary_when_successful() -> None:
    """FailoverLLMClient returns primary output and preserves provider when successful."""
    mock_primary = MagicMock(spec=LLMClient)
    mock_primary.provider = "llm_gemini"
    mock_primary.last_effective_provider = "llm_gemini"
    mock_primary.complete.return_value = "PRIMARY CLOUD RESPONSE"

    mock_fallback = MagicMock(spec=LLMClient)
    mock_fallback.provider = "llm_local"

    failover = FailoverLLMClient(
        provider="llm_gemini",
        primary=mock_primary,
        fallback=mock_fallback,
        has_api_key=True,
    )

    out = failover.complete(system="Sys", user="User")

    assert out == "PRIMARY CLOUD RESPONSE"
    assert failover.last_effective_provider == "llm_gemini"
    mock_fallback.complete.assert_not_called()


def test_build_client_factory_local() -> None:
    """build_client returns OpenAICompatibleClient for llm_local."""
    settings = make_test_settings()
    client = build_client(provider="llm_local", settings=settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "llm_local"
    assert client.model == "qwen2.5-3b"


def test_build_client_factory_cloud_with_and_without_keys() -> None:
    """build_client configures FailoverLLMClient for cloud backends."""
    # Without keys
    settings_no_keys = make_test_settings(gemini_key=None)
    client_gemini = build_client(provider="llm_gemini", settings=settings_no_keys)

    assert isinstance(client_gemini, FailoverLLMClient)
    assert client_gemini.has_api_key is False

    # With keys
    settings_with_keys = make_test_settings(gemini_key="real-gemini")
    client_gemini_k = build_client(provider="llm_gemini", settings=settings_with_keys)

    assert isinstance(client_gemini_k, FailoverLLMClient)
    assert client_gemini_k.has_api_key is True


def test_build_client_unknown_provider_defaults_to_local() -> None:
    """An unknown provider string falls back gracefully to local client."""
    settings = make_test_settings()
    client = build_client(provider="llm_unknown_custom", settings=settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "llm_local"


def test_openai_compatible_client_raises_llm_connection_error() -> None:
    """OpenAICompatibleClient converts connection refused into LLMConnectionError."""
    import openai

    from src.llm_client import LLMConnectionError

    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = openai.APIConnectionError(request=MagicMock())

    client = OpenAICompatibleClient(
        provider="llm_local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        model="qwen2.5-3b",
        client=mock_inner,
    )

    with pytest.raises(LLMConnectionError, match="Connection refused"):
        client.complete(system="Sys", user="User")


def test_openai_compatible_client_raises_llm_quota_error() -> None:
    """OpenAICompatibleClient converts rate limit errors into LLMQuotaError."""
    import openai

    from src.llm_client import LLMQuotaError

    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = openai.RateLimitError(
        message="Quota exceeded", response=MagicMock(), body={}
    )

    client = OpenAICompatibleClient(
        provider="llm_local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        model="qwen2.5-3b",
        client=mock_inner,
    )

    with pytest.raises(LLMQuotaError, match="Rate limit exceeded"):
        client.complete(system="Sys", user="User")


def test_failover_raises_connection_error_when_fallback_fails() -> None:
    """FailoverLLMClient raises LLMConnectionError when both primary and fallback fail."""
    from src.llm_client import LLMConnectionError

    mock_fallback = MagicMock(spec=LLMClient)
    mock_fallback.provider = "llm_local"
    mock_fallback.complete.side_effect = LLMConnectionError("Connection refused on 8080")

    failover = FailoverLLMClient(
        provider="llm_gemini",
        primary=None,
        fallback=mock_fallback,
        has_api_key=False,
    )

    with pytest.raises(LLMConnectionError, match="fallback to llm_local failed"):
        failover.complete(system="Sys", user="User")


def test_main_cli_execution(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """main CLI utility connects and prints response from mock client."""
    from src import llm_client

    class FakeClient:
        provider = "llm_local"
        last_effective_provider = "llm_local"

        def complete(self, system: str, user: str, schema: Any = None) -> str:
            return "OK TEST RESPONSE"

    monkeypatch.setattr(llm_client, "build_client", lambda *args, **kwargs: FakeClient())

    main(argv=["--provider", "llm_local", "--prompt", "Hello"])

    captured = capsys.readouterr().out
    assert "Effective provider: llm_local" in captured
    assert "Output: OK TEST RESPONSE" in captured


def test_main_cli_handles_connection_error_gracefully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main CLI utility catches LLMConnectionError and prints actionable advice to stderr."""
    from src import llm_client
    from src.llm_client import LLMConnectionError

    class FailingClient:
        provider = "llm_local"
        last_effective_provider = "llm_local"

        def complete(self, system: str, user: str, schema: Any = None) -> str:
            raise LLMConnectionError("Could not connect to llm_local at http://127.0.0.1:8080/v1")

    monkeypatch.setattr(llm_client, "build_client", lambda *args, **kwargs: FailingClient())

    main(argv=["--provider", "llm_local", "--prompt", "Hello"])

    captured = capsys.readouterr()
    assert "Connection Error: Could not connect to llm_local" in captured.err
    assert "run_llama_server.sh" in captured.err


def test_gemini_client_sanitizes_additional_properties_from_schema() -> None:
    """GeminiClient recursively removes additionalProperties to avoid HTTP 400."""
    captured_config: Any = None

    class FakeModels:
        def generate_content(self, model: str, contents: str, config: Any) -> Any:
            nonlocal captured_config
            captured_config = config
            mock_res = MagicMock()
            mock_res.text = '{"status": "ok"}'
            return mock_res

    fake_client = MagicMock()
    fake_client.models = FakeModels()

    client = GeminiClient(api_key="test-key", client=fake_client)
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }

    client.complete(system="test", user="test", schema=input_schema)

    assert captured_config is not None
    schema_dict = captured_config.response_schema
    assert "additionalProperties" not in schema_dict
    assert "additional_properties" not in schema_dict
    assert "additionalProperties" not in schema_dict["properties"]["nested"]
    assert "additional_properties" not in schema_dict["properties"]["nested"]
