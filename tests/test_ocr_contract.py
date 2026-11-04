"""Contract tests for EasyOCR and OcrEngine protocol compliance.

All standard tests run strictly offline, deterministically, and in milliseconds by injecting
mock reader objects. Live weights testing is quarantined under the 'slow' pytest marker.

Ref: Docs/implementation_plan.md, section 7, WP-05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.core.settings import load_settings
from src.core.types import BBox, OcrDocument, OcrEngine, OcrLine
from src.ocr.easyocr import EasyOCREngine, main


class MockReader:
    """Mock easyocr.Reader returning pre-configured text items."""

    def __init__(self, items: list[tuple[list[list[int]], str, float]]) -> None:
        self.items = items

    def readtext(self, image: np.ndarray, detail: int = 1) -> list[tuple[list[list[int]], str, float]]:
        return self.items


def test_easyocr_engine_satisfies_protocol() -> None:
    """EasyOCREngine satisfies the structural OcrEngine protocol."""
    mock = MockReader([])
    engine: OcrEngine = EasyOCREngine(reader=mock)

    assert engine.name == "easyocr"
    assert hasattr(engine, "read")


def test_reading_order_and_sequential_ids() -> None:
    """Lines returned out-of-order are sorted top-to-bottom and numbered 1..N without gaps."""
    # Bottom line first, top line second
    mock_items = [
        ([[10, 80], [100, 80], [100, 100], [10, 100]], "TOTAL 16.10", 0.95),
        ([[10, 10], [90, 10], [90, 30], [10, 30]], "PINGO DOCE", 0.98),
        ([[10, 40], [120, 40], [120, 60], [10, 60]], "NIF: 500100144", 0.92),
    ]
    engine = EasyOCREngine(reader=MockReader(mock_items))
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    doc = engine.read(image)

    assert isinstance(doc, OcrDocument)
    assert doc.engine == "easyocr"
    assert doc.image_size == (200, 120)
    assert len(doc.lines) == 3

    assert [line.id for line in doc.lines] == [1, 2, 3]
    assert doc.lines[0].text == "PINGO DOCE"
    assert doc.lines[1].text == "NIF: 500100144"
    assert doc.lines[2].text == "TOTAL 16.10"


def test_horizontal_fragments_merged_into_single_line() -> None:
    """Multi-column fragments on the same horizontal level are merged left-to-right."""
    mock_items = [
        ([[100, 50], [180, 50], [180, 70], [100, 70]], "16,10 EUR", 0.90),
        ([[10, 52], [80, 52], [80, 72], [10, 72]], "TOTAL", 0.95),
    ]
    engine = EasyOCREngine(reader=MockReader(mock_items))
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    doc = engine.read(image)

    assert len(doc.lines) == 1
    assert doc.lines[0].id == 1
    assert doc.lines[0].text == "TOTAL 16,10 EUR"
    assert doc.lines[0].bbox.x1 == 10
    assert doc.lines[0].bbox.x2 == 180
    assert doc.lines[0].confidence == pytest.approx(0.925)


def test_bbox_clamping_to_image_boundaries() -> None:
    """Coordinates extending outside image edges are clamped to image bounds."""
    mock_items = [
        ([[-20, -10], [250, -10], [250, 150], [-20, 150]], "OVERSIZED LINE", 0.85),
    ]
    engine = EasyOCREngine(reader=MockReader(mock_items))
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    doc = engine.read(image)

    assert len(doc.lines) == 1
    bbox = doc.lines[0].bbox
    assert bbox.x1 == 0
    assert bbox.y1 == 0
    assert bbox.x2 == 200
    assert bbox.y2 == 100
    assert bbox.width == 200
    assert bbox.height == 100


def test_filters_empty_text_and_zero_area_boxes() -> None:
    """Whitespace-only lines and degenerate zero-area boxes are skipped."""
    mock_items = [
        ([[10, 10], [50, 10], [50, 30], [10, 30]], "   ", 0.9),  # empty text
        ([[10, 10], [10, 10], [10, 10], [10, 10]], "VALID TEXT", 0.9),  # zero area
        ([[10, 40], [80, 40], [80, 60], [10, 60]], "KEEP ME", 0.91),  # valid
    ]
    engine = EasyOCREngine(reader=MockReader(mock_items))
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    doc = engine.read(image)

    assert len(doc.lines) == 1
    assert doc.lines[0].id == 1
    assert doc.lines[0].text == "KEEP ME"


def test_confidence_clamped_to_valid_range() -> None:
    """Confidence values are clamped to [0.0, 1.0]."""
    mock_items = [
        ([[10, 10], [50, 10], [50, 30], [10, 30]], "OVER", 1.5),
        ([[10, 40], [50, 40], [50, 60], [10, 60]], "UNDER", -0.2),
    ]
    engine = EasyOCREngine(reader=MockReader(mock_items))
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    doc = engine.read(image)

    assert doc.lines[0].confidence == 1.0
    assert doc.lines[1].confidence == 0.0


def test_ocr_document_prompt_formatting() -> None:
    """Prompt text output conforms to zero-padded [NN] citations."""
    lines = [
        OcrLine(id=1, text="LINE A", confidence=0.9, bbox=BBox(0, 0, 10, 10)),
        OcrLine(id=2, text="LINE B", confidence=0.8, bbox=BBox(0, 10, 10, 20)),
    ]
    doc = OcrDocument(lines=lines, engine="easyocr", elapsed_ms=10.0, image_size=(100, 100))

    assert doc.as_prompt_text() == "[01] LINE A\n[02] LINE B"


def test_empty_image_raises_value_error() -> None:
    """Empty or zero-sized image raises ValueError."""
    engine = EasyOCREngine(reader=MockReader([]))
    empty = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty or zero-sized"):
        engine.read(empty)


def test_main_cli_missing_or_empty_receipts_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints error when receipts directory is missing or empty."""
    settings = load_settings(repo_root=tmp_path)
    main(argv=[], settings_override=settings)
    assert "Receipts directory not found" in capsys.readouterr().out


def test_main_cli_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints error when explicitly specified file does not exist."""
    main(argv=["--file", str(tmp_path / "nonexistent.jpg")])
    assert "Specified file not found" in capsys.readouterr().out


def test_main_cli_execution_with_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main executes end-to-end when using an injected mock engine."""
    from src.ocr import easyocr

    img_path = tmp_path / "receipt.jpg"
    synthetic = np.full((120, 80, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_path), synthetic)

    mock_items = [
        ([[10, 10], [70, 10], [70, 30], [10, 30]], "MOCK RECEIPT TEXT", 0.99),
    ]

    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._real = EasyOCREngine(reader=MockReader(mock_items))

        def read(self, image: np.ndarray) -> OcrDocument:
            return self._real.read(image)

    monkeypatch.setattr(easyocr, "EasyOCREngine", FakeEngine)

    main(argv=["--file", str(img_path), "--no-preprocess"])

    captured = capsys.readouterr().out
    assert "MOCK RECEIPT TEXT" in captured
    assert "Lines found: 1" in captured


@pytest.mark.slow
def test_live_easyocr_engine_smoke() -> None:
    """Live EasyOCR model smoke test (opt-in only via `pytest -m slow`)."""
    engine = EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True)
    synthetic = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.putText(synthetic, "TEST 123", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    doc = engine.read(synthetic)

    assert isinstance(doc, OcrDocument)
    assert doc.engine == "easyocr"
    assert len(doc.lines) >= 1
