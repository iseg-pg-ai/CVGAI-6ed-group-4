"""Unit tests for the llama-server launcher module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.server_launcher import build_parser, find_llama_server_binary, main
from src.core.settings import Settings


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    """Create isolated settings for launcher tests."""
    model_file = tmp_path / "models" / "fake_model.gguf"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()

    return Settings(
        llama_base_url="http://127.0.0.1:8080/v1",
        llama_model="qwen2.5-3b",
        llama_timeout_s=30.0,
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash",
        ocr_engine="easyocr",
        easyocr_langs=("pt", "en"),
        tesseract_cmd=None,
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
        receipts_dir=tmp_path / "data" / "receipts",
        ground_truth_path=tmp_path / "data" / "ground_truth.csv",
        results_dir=tmp_path / "results",
        llama_port=8080,
        llama_ngl=0,
        llama_ctx=8192,
        llama_model_path=model_file,
    )


def test_build_parser_options() -> None:
    """Parser parses positional model and explicit flag options."""
    parser = build_parser()
    args = parser.parse_args(["custom_model.gguf", "--port", "9000", "-ngl", "33", "-c", "4096", "--dry-run"])

    assert args.model == "custom_model.gguf"
    assert args.port == 9000
    assert args.ngl == 33
    assert args.ctx == 4096
    assert args.dry_run is True


def test_find_llama_server_binary() -> None:
    """find_llama_server_binary returns candidate from shutil.which."""
    with patch("shutil.which", return_value="/mock/path/llama-server"):
        assert find_llama_server_binary() == "/mock/path/llama-server"

    with patch("shutil.which", return_value=None):
        assert find_llama_server_binary() is None


def test_main_dry_run_with_settings(fake_settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """main with --dry-run prints command and exits with 0."""
    code = main(["--dry-run"], settings_override=fake_settings)
    out = capsys.readouterr().out

    assert code == 0
    assert "Starting llama-server" in out
    assert "fake_model.gguf" in out
    assert "--port 8080" in out
    assert "-ngl 0" in out
    assert "-c 8192" in out


def test_main_missing_binary(
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main fails with status 1 when llama-server binary is not found."""
    with patch("src.core.server_launcher.find_llama_server_binary", return_value=None):
        code = main([], settings_override=fake_settings)

    err = capsys.readouterr().err
    assert code == 1
    assert "executable not found on system PATH" in err


def test_main_missing_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main fails with status 1 when model file does not exist."""
    missing_settings = Settings(
        llama_base_url="http://127.0.0.1:8080/v1",
        llama_model="qwen2.5-3b",
        llama_timeout_s=30.0,
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash",
        ocr_engine="easyocr",
        easyocr_langs=("pt", "en"),
        tesseract_cmd=None,
        repo_root=tmp_path,
        data_dir=tmp_path / "data",
        receipts_dir=tmp_path / "data" / "receipts",
        ground_truth_path=tmp_path / "data" / "ground_truth.csv",
        results_dir=tmp_path / "results",
        llama_model_path=tmp_path / "does_not_exist.gguf",
    )

    with patch("src.core.server_launcher.find_llama_server_binary", return_value="/bin/llama-server"):
        code = main([], settings_override=missing_settings)

    err = capsys.readouterr().err
    assert code == 1
    assert "Model file not found" in err


def test_main_keyboard_interrupt(fake_settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    """User interruption (Ctrl+C) returns 0 and prints notice."""
    with (
        patch("src.core.server_launcher.find_llama_server_binary", return_value="/bin/llama-server"),
        patch("subprocess.run", side_effect=KeyboardInterrupt),
    ):
        code = main([], settings_override=fake_settings)

    out = capsys.readouterr().out
    assert code == 0
    assert "llama-server stopped by user." in out
