"""Unit tests for the Gradio interactive application module.

All tests run completely offline and deterministically in milliseconds without
launching a live HTTP server, downloading external weights, or touching the real golden set.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-16.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.app import (
    _compute_cache_key,
    _from_gradio_rgb,
    _get_ocr_document,
    _get_ocr_engine,
    _to_gradio_rgb,
    answer_question,
    build_app,
    clear_caches,
    load_benchmark_report,
    process_document,
)
from src.preprocessing import PreprocessOptions
from src.settings import Settings
from src.types import BBox, OcrDocument, OcrLine


@pytest.fixture(autouse=True)
def clean_app_caches() -> Any:
    """Ensure in-memory caches are isolated across tests."""
    clear_caches()
    yield
    clear_caches()


@pytest.fixture
def mock_app_settings(tmp_path: Path) -> Settings:
    """Create isolated settings pointing to an empty tmp_path directory."""
    data_dir = tmp_path / "data"
    receipts_dir = data_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        llama_base_url="http://127.0.0.1:8080/v1",
        llama_model="test-model",
        llama_timeout_s=30.0,
        gemini_api_key=None,
        gemini_model="gemini",
        ocr_engine="easyocr",
        easyocr_langs=("pt", "en"),
        tesseract_cmd=None,
        repo_root=tmp_path,
        data_dir=data_dir,
        receipts_dir=receipts_dir,
        ground_truth_path=data_dir / "ground_truth.csv",
        results_dir=results_dir,
    )


def test_build_app_instantiates_blocks_with_isolated_settings(mock_app_settings: Settings) -> None:
    """build_app() constructs Gradio Blocks without touching real repository images."""
    demo = build_app(settings_override=mock_app_settings)
    assert demo is not None
    assert hasattr(demo, "queue")
    assert hasattr(demo, "launch")


def test_color_space_conversions() -> None:
    """Color conversion helpers faithfully map between Gradio RGB and OpenCV BGR."""
    # 3-channel RGB where Red=255, Green=10, Blue=5
    rgb_in = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_in[:, :, 0] = 255
    rgb_in[:, :, 1] = 10
    rgb_in[:, :, 2] = 5

    bgr = _from_gradio_rgb(rgb_in)
    assert bgr.shape == (2, 2, 3)
    assert bgr[0, 0, 0] == 5  # B
    assert bgr[0, 0, 1] == 10  # G
    assert bgr[0, 0, 2] == 255  # R

    rgb_out = _to_gradio_rgb(bgr)
    assert rgb_out is not None
    np.testing.assert_array_equal(rgb_out, rgb_in)

    # 4-channel RGBA conversion
    rgba_in = np.full((2, 2, 4), 100, dtype=np.uint8)
    bgr_from_rgba = _from_gradio_rgb(rgba_in)
    assert bgr_from_rgba.shape == (2, 2, 3)

    # Grayscale conversion
    gray_in = np.full((2, 2), 50, dtype=np.uint8)
    bgr_from_gray = _from_gradio_rgb(gray_in)
    assert bgr_from_gray.shape == (2, 2, 3)

    # Unsupported shape raises
    with pytest.raises(ValueError, match="Formato de imagem"):
        _from_gradio_rgb(np.zeros((2, 2, 5), dtype=np.uint8))

    assert _to_gradio_rgb(None) is None


def test_compute_cache_key_deterministic_and_sensitive() -> None:
    """_compute_cache_key produces identical hashes for identical inputs and distinguishes shapes."""
    img1 = np.full((10, 20, 3), 128, dtype=np.uint8)
    img2 = np.full((20, 10, 3), 128, dtype=np.uint8)  # same byte count, different shape
    opts1 = PreprocessOptions(rotation_deg="auto", auto_crop=True)
    opts2 = PreprocessOptions(rotation_deg="auto", auto_crop=True)
    opts3 = PreprocessOptions(rotation_deg=90, auto_crop=False)

    key1 = _compute_cache_key(img1, opts1, "easyocr")
    key1_again = _compute_cache_key(img1, opts2, "easyocr")
    key2 = _compute_cache_key(img2, opts1, "easyocr")
    key3 = _compute_cache_key(img1, opts3, "easyocr")

    assert key1 == key1_again
    assert key1 != key2
    assert key1 != key3


def test_engine_cache_and_document_cache_hit(monkeypatch: pytest.MonkeyPatch, mock_app_settings: Settings) -> None:
    """_get_ocr_document reuses cached OCR document and engine on repeated requests."""
    read_calls = 0

    class CountingEngine:
        name = "easyocr"

        def read(self, image: np.ndarray) -> OcrDocument:
            nonlocal read_calls
            read_calls += 1
            return OcrDocument(
                lines=[OcrLine(id=1, text="TEST", confidence=0.9, bbox=BBox(0, 0, 10, 10))],
                engine="easyocr",
                elapsed_ms=1.0,
                image_size=(image.shape[1], image.shape[0]),
            )

    monkeypatch.setattr("src.app.EasyOCREngine", lambda *args, **kwargs: CountingEngine())

    engine1 = _get_ocr_engine("EasyOCR", settings=mock_app_settings)
    engine2 = _get_ocr_engine("EasyOCR", settings=mock_app_settings)
    assert engine1 is engine2

    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    opts = PreprocessOptions(rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0)

    # First call: cache miss
    _, doc1, _ = _get_ocr_document(img, opts, "EasyOCR", settings=mock_app_settings)
    assert read_calls == 1

    # Second call: cache hit, read() must not be called again
    _, doc2, _ = _get_ocr_document(img, opts, "EasyOCR", settings=mock_app_settings)
    assert read_calls == 1
    assert doc1 is doc2


def test_process_document_handles_none_and_invalid_image(mock_app_settings: Settings) -> None:
    """process_document returns empty placeholders when input image is None or malformed."""
    res_none = process_document(
        image=None,
        rotation_str="auto",
        auto_crop=True,
        remove_shadows=True,
        clahe_clip=2.0,
        ocr_engine_name="EasyOCR",
        extractor_name="Regex Baseline",
        settings=mock_app_settings,
    )
    assert res_none[0] is None
    assert "Nenhuma imagem" in res_none[2]
    assert res_none[3] == []

    # Invalid image shape
    bad_img = np.zeros((10, 10, 5), dtype=np.uint8)
    res_bad = process_document(
        image=bad_img,
        rotation_str="auto",
        auto_crop=True,
        remove_shadows=True,
        clahe_clip=2.0,
        ocr_engine_name="EasyOCR",
        extractor_name="Regex Baseline",
        settings=mock_app_settings,
    )
    assert "Erro na conversao" in res_bad[2]


def test_process_document_execution_and_rgb_output(
    monkeypatch: pytest.MonkeyPatch, mock_app_settings: Settings
) -> None:
    """process_document returns RGB arrays ready for Gradio browser rendering."""
    mock_doc = OcrDocument(
        lines=[
            OcrLine(id=1, text="PINGO DOCE", confidence=0.98, bbox=BBox(10, 10, 90, 30)),
            OcrLine(id=2, text="NIF: 500100144", confidence=0.95, bbox=BBox(10, 35, 90, 50)),
            OcrLine(id=3, text="Total: 16.10 EUR", confidence=0.99, bbox=BBox(10, 55, 90, 75)),
        ],
        engine="easyocr",
        elapsed_ms=10.0,
        image_size=(100, 100),
    )

    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return mock_doc

    monkeypatch.setattr("src.app.EasyOCREngine", FakeEngine)
    monkeypatch.setattr("src.app.read_qr", lambda *args, **kwargs: {"total": "16.10", "nif_comerciante": "500100144"})

    synthetic_rgb = np.full((100, 100, 3), 220, dtype=np.uint8)

    (
        annotated,
        crop_img,
        info,
        table,
        step1,
        step2,
        step3,
        step4,
        prompt_txt,
    ) = process_document(
        image=synthetic_rgb,
        rotation_str="0",
        auto_crop=False,
        remove_shadows=False,
        clahe_clip=0.0,
        ocr_engine_name="EasyOCR",
        extractor_name="Regex Baseline",
        settings=mock_app_settings,
    )

    assert annotated is not None
    assert crop_img is not None
    assert info != ""
    assert len(table) == 5
    field_names = [row[0] for row in table]
    assert field_names == ["entidade", "nif_comerciante", "nif_cliente", "data", "total"]

    # Verify returned steps
    assert step1 is not None
    assert step4 is not None
    assert "[01] PINGO DOCE" in prompt_txt


def test_process_document_llm_extractor_fallback(monkeypatch: pytest.MonkeyPatch, mock_app_settings: Settings) -> None:
    """When an LLM extractor fails, process_document falls back to regex with an explanatory note."""
    mock_doc = OcrDocument(
        lines=[OcrLine(id=1, text="Total 16.10", confidence=0.99, bbox=BBox(0, 0, 10, 10))],
        engine="easyocr",
        elapsed_ms=1.0,
        image_size=(20, 20),
    )

    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return mock_doc

    monkeypatch.setattr("src.app.EasyOCREngine", FakeEngine)

    def failing_llm(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llama-server unreachable on 8080")

    monkeypatch.setattr("src.app.extract_llm", failing_llm)

    synthetic_rgb = np.full((50, 50, 3), 200, dtype=np.uint8)
    res = process_document(
        image=synthetic_rgb,
        rotation_str="0",
        auto_crop=False,
        remove_shadows=False,
        clahe_clip=0.0,
        ocr_engine_name="EasyOCR",
        extractor_name="LLM Local (llama.cpp)",
        settings=mock_app_settings,
    )

    info = res[2]
    assert "Aviso: LLM Local (llama.cpp) indisponivel" in info
    assert "recurso alternativo" in info


def test_answer_question_validation_and_error_handling(mock_app_settings: Settings) -> None:
    """answer_question validates missing inputs and catches processing exceptions."""
    status1, ans1, _, _ = answer_question(
        "", None, "auto", True, True, 2.0, "EasyOCR", "LLM Local (llama.cpp)", settings=mock_app_settings
    )
    assert status1 == "AUSENTE NO DOCUMENTO"
    assert "Nenhum documento" in ans1

    dummy_img = np.full((50, 50, 3), 200, dtype=np.uint8)
    status2, ans2, _, _ = answer_question(
        "   ", dummy_img, "auto", True, True, 2.0, "EasyOCR", "LLM Local (llama.cpp)", settings=mock_app_settings
    )
    assert status2 == "AUSENTE NO DOCUMENTO"
    assert "Por favor introduza" in ans2

    # Malformed array
    status3, ans3, _, _ = answer_question(
        "Qual o total?",
        np.zeros((10, 10, 5), dtype=np.uint8),
        "auto",
        True,
        True,
        2.0,
        "EasyOCR",
        "LLM Local (llama.cpp)",
        settings=mock_app_settings,
    )
    assert status3 == "AUSENTE NO DOCUMENTO"
    assert "Erro na conversao" in ans3


def test_answer_question_with_mock_qa(monkeypatch: pytest.MonkeyPatch, mock_app_settings: Settings) -> None:
    """answer_question queries QA assistant and produces grounded visual highlights."""
    from src import app
    from src.types import Answer

    mock_doc = OcrDocument(
        lines=[OcrLine(id=1, text="Total 16.10", confidence=0.99, bbox=BBox(10, 10, 50, 30))],
        engine="easyocr",
        elapsed_ms=5.0,
        image_size=(60, 60),
    )

    monkeypatch.setattr(
        app,
        "_get_ocr_document",
        lambda *args, **kwargs: (MagicMock(image=np.zeros((60, 60, 3), dtype=np.uint8)), mock_doc, None),
    )

    mock_ans = Answer(
        question="Qual o total?",
        answer="O total e 16.10 EUR.",
        status="ENCONTRADO",
        line_ids=[1],
        provider="llm_local",
    )
    monkeypatch.setattr("src.app.ask_qa", lambda *args, **kwargs: mock_ans)

    dummy_rgb = np.full((60, 60, 3), 255, dtype=np.uint8)
    status, text, annotated, crop_res = answer_question(
        question="Qual o total?",
        image=dummy_rgb,
        rotation_str="0",
        auto_crop=False,
        remove_shadows=False,
        clahe_clip=0.0,
        ocr_engine_name="EasyOCR",
        provider_choice="LLM Local (llama.cpp)",
        settings=mock_app_settings,
    )

    assert status == "ENCONTRADO"
    assert "16.10 EUR" in text
    assert annotated is not None
    assert crop_res is not None


def test_load_benchmark_report_fallback_and_existing(mock_app_settings: Settings) -> None:
    """load_benchmark_report provides instructions when files are missing and loads when present."""
    # When missing
    md, headers, rows = load_benchmark_report(settings=mock_app_settings)
    assert "Relatorio de Avaliacao" in md
    assert headers == []
    assert rows == []

    # When present
    (mock_app_settings.results_dir / "metrics.md").write_text("# Real Benchmark Metrics\nTest table", encoding="utf-8")
    (mock_app_settings.results_dir / "metrics.csv").write_text("campo,taxa_acerto\ntotal,1.0\n", encoding="utf-8")

    md2, headers2, rows2 = load_benchmark_report(settings=mock_app_settings)
    assert "Real Benchmark Metrics" in md2
    assert headers2 == ["campo", "taxa_acerto"]
    assert rows2 == [["total", "1.0"]]
