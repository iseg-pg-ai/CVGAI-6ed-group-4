"""Unit tests for the application settings module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import Settings, get_settings, load_settings


def test_default_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load with documented defaults when no environment variables are set."""
    # Clear any project-specific env vars that might exist in developer shells
    for var in [
        "LLAMA_BASE_URL",
        "LLAMA_MODEL",
        "LLAMA_TIMEOUT_S",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "OCR_ENGINE",
        "EASYOCR_LANGS",
        "TESSERACT_CMD",
        "DATA_DIR",
        "RECEIPTS_DIR",
        "GROUND_TRUTH_PATH",
        "RESULTS_DIR",
        "LLAMA_MODEL_PATH",
        "MODEL",
        "LLAMA_PORT",
        "PORT",
        "LLAMA_NGL",
        "NGL",
        "LLAMA_CTX",
        "CTX",
        "PREPROCESS_MAX_SIDE",
        "PREPROCESS_CLAHE_CLIP",
        "PREPROCESS_GAMMA",
        "PREPROCESS_SHARPEN",
    ]:
        monkeypatch.delenv(var, raising=False)

    settings = load_settings(repo_root=tmp_path)

    assert settings.llama_base_url == "http://127.0.0.1:8080/v1"
    assert settings.llama_model == "local-llm"
    assert settings.llama_timeout_s == 120.0
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-3.5-flash"
    assert settings.ocr_engine == "easyocr"
    assert settings.easyocr_langs == ("pt", "en")
    assert settings.tesseract_cmd is None
    assert settings.repo_root == tmp_path.resolve()
    assert settings.data_dir == (tmp_path / "data").resolve()
    assert settings.receipts_dir == (tmp_path / "data" / "receipts").resolve()
    assert settings.ground_truth_path == (tmp_path / "data" / "ground_truth.csv").resolve()
    assert settings.results_dir == (tmp_path / "results").resolve()
    assert settings.llama_port == 8080
    assert settings.llama_ngl == 0
    assert settings.llama_ctx == 8192
    assert settings.llama_model_path == (tmp_path / "models" / "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf").resolve()


def test_empty_string_api_keys_normalize_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty strings in environment or .env are normalized to None."""
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setenv("TESSERACT_CMD", "")

    settings = load_settings(repo_root=tmp_path)

    assert settings.gemini_api_key is None
    assert settings.tesseract_cmd is None


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables successfully override defaults."""
    monkeypatch.setenv("LLAMA_BASE_URL", "http://192.168.1.10:8080/v1")
    monkeypatch.setenv("LLAMA_MODEL", "qwen2.5-3b")
    monkeypatch.setenv("LLAMA_TIMEOUT_S", "45.5")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setenv("EASYOCR_LANGS", "pt, es, fr")
    monkeypatch.setenv("TESSERACT_CMD", "/usr/bin/tesseract")
    monkeypatch.setenv("DATA_DIR", "custom_data")
    monkeypatch.setenv("LLAMA_PORT", "9090")
    monkeypatch.setenv("LLAMA_NGL", "32")
    monkeypatch.setenv("LLAMA_CTX", "4096")
    monkeypatch.setenv("LLAMA_MODEL_PATH", "custom_models/model.gguf")

    settings = load_settings(repo_root=tmp_path)

    assert settings.llama_base_url == "http://192.168.1.10:8080/v1"
    assert settings.llama_model == "qwen2.5-3b"
    assert settings.llama_timeout_s == 45.5
    assert settings.gemini_api_key == "secret-gemini-key"
    assert settings.ocr_engine == "tesseract"
    assert settings.easyocr_langs == ("pt", "es", "fr")
    assert settings.tesseract_cmd == Path("/usr/bin/tesseract").resolve()
    assert settings.data_dir == (tmp_path / "custom_data").resolve()
    assert settings.llama_port == 9090
    assert settings.llama_ngl == 32
    assert settings.llama_ctx == 4096
    assert settings.llama_model_path == (tmp_path / "custom_models/model.gguf").resolve()


def test_invalid_timeout_falls_back_to_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed numeric timeout defaults safely to 120.0 instead of crashing."""
    monkeypatch.setenv("LLAMA_TIMEOUT_S", "not-a-number")

    settings = load_settings(repo_root=tmp_path)

    assert settings.llama_timeout_s == 120.0


def test_get_settings_returns_cached_instance() -> None:
    """get_settings returns a valid Settings instance and caches it."""
    s1 = get_settings()
    s2 = get_settings()

    assert isinstance(s1, Settings)
    assert s1 is s2


def test_settings_supports_dataclasses_replace() -> None:
    """Settings supports dataclasses.replace for backward compatibility."""
    from dataclasses import replace

    s = get_settings()
    s_new = replace(s, llama_port=7777)
    assert s_new.llama_port == 7777
    assert s.llama_port != 7777
