"""Unit tests for the visual evidence highlighting and cropping module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.evidence import crop, draw_all_ocr_boxes, highlight, main
from src.settings import load_settings
from src.types import BBox, OcrDocument, OcrLine


@pytest.fixture
def sample_ocr_doc() -> OcrDocument:
    """Create a sample OcrDocument with known geometries for visual testing."""
    lines = [
        OcrLine(id=1, text="PINGO DOCE", confidence=0.98, bbox=BBox(x1=20, y1=10, x2=120, y2=40)),
        OcrLine(id=2, text="NIF: 500100144", confidence=0.95, bbox=BBox(x1=20, y1=50, x2=140, y2=70)),
        OcrLine(id=3, text="TOTAL 16.10", confidence=0.99, bbox=BBox(x1=20, y1=120, x2=130, y2=150)),
    ]
    return OcrDocument(lines=lines, engine="easyocr", elapsed_ms=5.0, image_size=(200, 200))


def test_draw_all_ocr_boxes_contract(sample_ocr_doc: OcrDocument) -> None:
    """draw_all_ocr_boxes returns a modified copy and does not mutate the source image."""
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
    original_copy = canvas.copy()

    rendered = draw_all_ocr_boxes(canvas, sample_ocr_doc)

    # Input image remains unmutated
    np.testing.assert_array_equal(canvas, original_copy)

    # Output has same shape and dtype
    assert rendered.shape == canvas.shape
    assert rendered.dtype == canvas.dtype

    # Output has drawing markings (not identical to plain white canvas)
    assert not np.array_equal(rendered, canvas)


def test_highlight_target_lines(sample_ocr_doc: OcrDocument) -> None:
    """highlight marks active lines with overlay while muting inactive lines."""
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
    original_copy = canvas.copy()

    highlighted = highlight(canvas, sample_ocr_doc, line_ids=[3], highlight_color=(0, 200, 255))

    # Non-destructive
    np.testing.assert_array_equal(canvas, original_copy)
    assert highlighted.shape == canvas.shape

    # Line 3 area (y=120..150, x=20..130) has been tinted by highlight_color
    line3_center_pixel = highlighted[135, 75]
    # In BGR, highlight_color has B=0, G=200, R=255; pure white was 255,255,255
    assert line3_center_pixel[0] < 255  # Blue channel reduced due to blending


def test_highlight_empty_or_unknown_line_ids(sample_ocr_doc: OcrDocument) -> None:
    """highlight handles empty or out-of-range line IDs gracefully."""
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)

    # No lines highlighted
    res1 = highlight(canvas, sample_ocr_doc, line_ids=[])
    assert res1.shape == canvas.shape

    # Out of range ID
    res2 = highlight(canvas, sample_ocr_doc, line_ids=[999])
    assert res2.shape == canvas.shape


def test_crop_single_line_dimensions(sample_ocr_doc: OcrDocument) -> None:
    """crop extracts single line with exact requested padding."""
    canvas = np.full((200, 200, 3), 128, dtype=np.uint8)
    pad = 10

    # Line 1 bbox: x1=20, y1=10, x2=120, y2=40 -> width=100, height=30
    snippet = crop(canvas, sample_ocr_doc, line_ids=[1], pad_px=pad)

    assert snippet is not None
    expected_w = 100 + (2 * pad)
    expected_h = 30 + (2 * pad)
    assert snippet.shape == (expected_h, expected_w, 3)


def test_crop_multi_line_union(sample_ocr_doc: OcrDocument) -> None:
    """crop computes union bounding box spanning multiple cited lines."""
    canvas = np.full((200, 200, 3), 128, dtype=np.uint8)
    pad = 5

    # Line 1: y1=10, y2=40, x1=20, x2=120
    # Line 3: y1=120, y2=150, x1=20, x2=130
    # Union bbox: x1=20, y1=10, x2=130, y2=150 -> w=110, h=140
    snippet = crop(canvas, sample_ocr_doc, line_ids=[1, 3], pad_px=pad)

    assert snippet is not None
    assert snippet.shape == (140 + (2 * pad), 110 + (2 * pad), 3)


def test_crop_empty_or_unknown_ids_returns_none(sample_ocr_doc: OcrDocument) -> None:
    """crop returns None when line_ids is empty or has no matches."""
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)

    assert crop(canvas, sample_ocr_doc, line_ids=[]) is None
    assert crop(canvas, sample_ocr_doc, line_ids=[999]) is None


def test_crop_clamps_to_image_boundaries() -> None:
    """Padding near image borders is clamped without indexing errors."""
    lines = [OcrLine(id=1, text="EDGE", confidence=1.0, bbox=BBox(x1=0, y1=0, x2=30, y2=20))]
    doc = OcrDocument(lines=lines, engine="easyocr", elapsed_ms=1.0, image_size=(100, 100))
    canvas = np.full((100, 100, 3), 200, dtype=np.uint8)

    snippet = crop(canvas, doc, line_ids=[1], pad_px=20)

    assert snippet is not None
    # x1=0, y1=0 clamped at 0; x2=30+20=50, y2=20+20=40
    assert snippet.shape == (40, 50, 3)


def test_empty_image_raises_value_error(sample_ocr_doc: OcrDocument) -> None:
    """Visual functions raise ValueError when given an empty image."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty or zero-sized"):
        draw_all_ocr_boxes(empty, sample_ocr_doc)

    with pytest.raises(ValueError, match="empty or zero-sized"):
        highlight(empty, sample_ocr_doc, line_ids=[1])

    with pytest.raises(ValueError, match="empty or zero-sized"):
        crop(empty, sample_ocr_doc, line_ids=[1])


def test_main_cli_missing_or_empty_receipts_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints error when receipts directory is missing or empty."""
    settings = load_settings(repo_root=tmp_path)
    main(argv=[], settings_override=settings)
    assert "Receipts directory not found" in capsys.readouterr().out


def test_main_cli_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints error when specified file does not exist."""
    main(argv=["--file", str(tmp_path / "nonexistent.jpg")])
    assert "Specified file not found" in capsys.readouterr().out


def test_main_cli_execution_and_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_ocr_doc: OcrDocument
) -> None:
    """main runs OCR and saves highlighted preview image to disk with --save."""
    img_path = tmp_path / "receipt.jpg"
    out_path = tmp_path / "highlighted.png"
    canvas = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(img_path), canvas)

    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return sample_ocr_doc

    monkeypatch.setattr("src.ocr_engine.EasyOCREngine", FakeEngine)

    main(argv=["--file", str(img_path), "--lines", "1,2", "--save", str(out_path)])

    assert out_path.is_file()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape == canvas.shape
