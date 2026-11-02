"""Offline contract tests for the Tesseract OCR backend."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from src.ocr_tesseract import TesseractEngine, main
from src.types import BBox, OcrDocument, OcrEngine, OcrLine


def _tesseract_data(
    text: list[str],
    conf: list[str],
    left: list[str],
    top: list[str],
    width: list[str],
    height: list[str],
    block_num: list[str],
    par_num: list[str],
    line_num: list[str],
) -> dict[str, list[Any]]:
    return {
        "text": text,
        "conf": conf,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "block_num": block_num,
        "par_num": par_num,
        "line_num": line_num,
    }


def test_tesseract_engine_satisfies_protocol() -> None:
    engine: OcrEngine = TesseractEngine()

    assert engine.name == "tesseract"
    assert hasattr(engine, "read")


def test_read_groups_words_and_sorts_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _tesseract_data(
        text=["TOTAL", "16,10", "NIF:", "501964843", "ignored"],
        conf=["90", "80", "70", "60", "-1"],
        left=["50", "110", "10", "50", "0"],
        top=["80", "80", "10", "10", "0"],
        width=["50", "40", "30", "80", "0"],
        height=["20", "20", "15", "15", "0"],
        block_num=["1", "1", "1", "1", "1"],
        par_num=["1", "1", "1", "1", "1"],
        line_num=["2", "2", "1", "1", "3"],
    )
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def fake_image_to_data(image: np.ndarray, **kwargs: object) -> dict[str, list[Any]]:
        calls.append((image, kwargs))
        return data

    monkeypatch.setattr("src.ocr_tesseract.pytesseract.image_to_data", fake_image_to_data)
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    doc = TesseractEngine().read(image)

    assert isinstance(doc, OcrDocument)
    assert doc.engine == "tesseract"
    assert doc.image_size == (200, 120)
    assert [line.id for line in doc.lines] == [1, 2]
    assert [line.text for line in doc.lines] == ["NIF: 501964843", "TOTAL 16,10"]
    assert doc.lines[0].confidence == pytest.approx(0.65)
    assert doc.lines[0].bbox.x1 == 10
    assert doc.lines[0].bbox.y1 == 10
    assert doc.lines[0].bbox.x2 == 130
    assert doc.lines[0].bbox.y2 == 25
    assert calls[0][0][0, 0].tolist() == [0, 0, 255]
    assert calls[0][1] == {
        "lang": "por",
        "config": "--psm 6",
        "output_type": "dict",
    }


def test_read_clamps_bounding_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _tesseract_data(
        text=["edge"],
        conf=["100"],
        left=["-10"],
        top=["-5"],
        width=["40"],
        height=["30"],
        block_num=["1"],
        par_num=["1"],
        line_num=["1"],
    )
    monkeypatch.setattr("src.ocr_tesseract.pytesseract.image_to_data", lambda *_args, **_kwargs: data)

    line = TesseractEngine().read(np.zeros((20, 20), dtype=np.uint8)).lines[0]

    assert (line.bbox.x1, line.bbox.y1, line.bbox.x2, line.bbox.y2) == (0, 0, 20, 20)
    assert line.confidence == 1.0


def test_read_uses_custom_executable_and_restores_global_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pytesseract

    original_cmd = pytesseract.pytesseract.tesseract_cmd
    data = _tesseract_data([], [], [], [], [], [], [], [], [])
    received_cmd: list[str] = []

    def fake_image_to_data(*_args: object, **_kwargs: object) -> dict[str, list[Any]]:
        received_cmd.append(pytesseract.pytesseract.tesseract_cmd)
        return data

    monkeypatch.setattr("src.ocr_tesseract.pytesseract.image_to_data", fake_image_to_data)

    result = TesseractEngine(tesseract_cmd=tmp_path / "tesseract.exe").read(np.zeros((10, 10), dtype=np.uint8))

    assert result.lines == []
    assert received_cmd == [str(tmp_path / "tesseract.exe")]
    assert pytesseract.pytesseract.tesseract_cmd == original_cmd


@pytest.mark.parametrize(
    ("lang", "psm"),
    [
        ("", 6),
        ("por", -1),
        ("por", 14),
    ],
)
def test_constructor_rejects_invalid_configuration(lang: str, psm: int) -> None:
    with pytest.raises(ValueError):
        TesseractEngine(lang=lang, psm=psm)


def test_constructor_falls_back_to_settings_tesseract_cmd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitting tesseract_cmd must use the executable configured in TESSERACT_CMD."""
    configured = tmp_path / "tesseract.exe"
    monkeypatch.setattr("src.settings.get_settings", lambda: SimpleNamespace(tesseract_cmd=configured))

    engine = TesseractEngine()

    assert engine.tesseract_cmd == str(configured)


def test_main_cli_missing_receipts_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main prints an error when the receipts directory is missing."""
    from src.settings import load_settings

    settings = load_settings(repo_root=tmp_path)
    main(argv=[], settings_override=settings)

    assert "Receipts directory not found" in capsys.readouterr().out


def test_main_cli_file_not_found(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main prints an error when the requested file does not exist."""
    main(argv=["--file", str(tmp_path / "nonexistent.jpg")])

    assert "Specified file not found" in capsys.readouterr().out


def test_main_cli_execution_with_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main runs end-to-end when the Tesseract engine is replaced by a fake."""
    from src import ocr_tesseract

    img_path = tmp_path / "receipt.jpg"
    cv2.imwrite(str(img_path), np.full((60, 120, 3), 200, dtype=np.uint8))

    class FakeEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            lines = [OcrLine(id=1, text="TOTAL 16,10", confidence=0.9, bbox=BBox(0, 0, 10, 10))]
            return OcrDocument(lines=lines, engine="tesseract", elapsed_ms=1.0, image_size=(120, 60))

    monkeypatch.setattr(ocr_tesseract, "TesseractEngine", FakeEngine)

    main(argv=["--file", str(img_path), "--no-preprocess"])

    captured = capsys.readouterr().out
    assert "TOTAL 16,10" in captured
    assert "Lines found: 1" in captured


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
    ],
)
def test_read_rejects_empty_or_unsupported_images(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        TesseractEngine().read(image)


def test_read_rejects_malformed_tesseract_output(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = _tesseract_data(
        text=["word"],
        conf=[],
        left=["0"],
        top=["0"],
        width=["5"],
        height=["5"],
        block_num=["1"],
        par_num=["1"],
        line_num=["1"],
    )
    monkeypatch.setattr("src.ocr_tesseract.pytesseract.image_to_data", lambda *_args, **_kwargs: malformed)

    with pytest.raises(ValueError, match="inconsistent lengths"):
        TesseractEngine().read(np.zeros((10, 10), dtype=np.uint8))
