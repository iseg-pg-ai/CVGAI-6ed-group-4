"""Unit and contract tests for the AT QR code reader and parser.

All tests run completely offline, deterministically, and fast using synthetic QR codes
generated via cv2.QRCodeEncoder or injected mock detectors.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-07.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.core.settings import load_settings
from src.vision.qr_reader import main, parse_at_qr_payload, read


def test_parse_at_qr_payload_full_valid_string() -> None:
    """Parses a standard Portaria 195/2020 payload with canonical aliases."""
    payload = (
        "A:518396894*B:261594842*C:PT*D:FS*E:N*F:20261024*G:FS 2/104504*"
        "H:JJ9PYK7K-104504*I1:PT*I5:12.74*N:1.98*O:16.10*Q:Yu48*R:1522"
    )
    result = parse_at_qr_payload(payload)

    assert result is not None
    assert result["A"] == "518396894"
    assert result["nif_comerciante"] == "518396894"
    assert result["B"] == "261594842"
    assert result["nif_cliente"] == "261594842"
    assert result["F"] == "20261024"
    assert result["data"] == "2026-10-24"
    assert result["O"] == "16.10"
    assert result["total"] == "16.10"
    assert result["atcud"] == "JJ9PYK7K-104504"
    assert result["tipo_doc"] == "FS"


def test_parse_at_qr_payload_consumidor_final() -> None:
    """Customer NIF 999999990 is mapped to empty string (Consumidor Final)."""
    payload = "A:500829993*B:999999990*D:FS*F:20261027*O:4.70"
    result = parse_at_qr_payload(payload)

    assert result is not None
    assert result["B"] == "999999990"
    assert result["nif_cliente"] == ""


def test_parse_at_qr_payload_rejects_non_at_strings() -> None:
    """Non-AT URLs, random strings, and payloads lacking A or O are rejected."""
    assert parse_at_qr_payload("https://www.pingo-doce.pt") is None
    assert parse_at_qr_payload("") is None
    assert parse_at_qr_payload("X:123*Y:456") is None
    assert parse_at_qr_payload("plain text with no asterisks") is None


def test_read_detects_and_decodes_synthetic_qr_code() -> None:
    """read() successfully decodes an in-memory generated synthetic AT QR code."""
    encoder = cv2.QRCodeEncoder.create()
    payload = "A:500829993*B:999999990*D:FS*F:20261027*O:5.86*H:ABC123"
    raw_matrix = encoder.encode(payload)

    # Scale and add border for standard detector readability
    scaled = cv2.resize(raw_matrix, (raw_matrix.shape[1] * 8, raw_matrix.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(scaled, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

    result = read(bgr)

    assert result is not None
    assert result["nif_comerciante"] == "500829993"
    assert result["total"] == "5.86"
    assert result["data"] == "2026-10-27"
    assert result["atcud"] == "ABC123"


def test_read_returns_none_on_blank_image() -> None:
    """An image with no QR code returns None without error."""
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert read(blank) is None


def test_read_empty_image_raises_value_error() -> None:
    """Empty or zero-sized image raises ValueError."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty or zero-sized"):
        read(empty)


def test_read_with_mock_detector_injection() -> None:
    """read() accepts custom detector instances for testing."""

    class MockDetector:
        def detectAndDecode(self, image: np.ndarray) -> tuple[str, Any, Any]:
            return "A:518396894*O:12.50", None, None

    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    result = read(img, detector=MockDetector())

    assert result is not None
    assert result["nif_comerciante"] == "518396894"
    assert result["total"] == "12.50"


def test_read_ignores_promotional_qr_and_decodes_at_code() -> None:
    """read() skips promotional non-AT QR codes and captures the fiscal AT QR code."""

    class MultiMockDetector:
        def detectAndDecodeMulti(self, image: np.ndarray) -> tuple[bool, list[str], Any, Any]:
            return True, ["https://continente.pt/app", "A:518396894*O:12.50"], None, None

    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    result = read(img, detector=MultiMockDetector())

    assert result is not None
    assert result["nif_comerciante"] == "518396894"
    assert result["total"] == "12.50"


def test_main_cli_missing_or_empty_receipts_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints friendly error when receipts directory is missing or empty."""
    settings = load_settings(repo_root=tmp_path)
    main(argv=[], settings_override=settings)
    assert "Receipts directory not found" in capsys.readouterr().out


def test_main_cli_file_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main prints error when specified file does not exist."""
    main(argv=["--file", str(tmp_path / "nonexistent.jpg")])
    assert "Specified file not found" in capsys.readouterr().out


def test_main_cli_execution_with_qr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main detects and prints AT fields on an image containing a QR code."""
    img_path = tmp_path / "qr_receipt.jpg"
    synthetic = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), synthetic)

    class MockDetector:
        def detectAndDecode(self, image: np.ndarray) -> tuple[str, Any, Any]:
            return "A:518396894*O:16.10*F:20261024", None, None

    monkeypatch.setattr(cv2, "QRCodeDetector", MockDetector)

    main(argv=["--file", str(img_path), "--no-preprocess"])

    captured = capsys.readouterr().out
    assert "AT Tax Authority QR Data Found" in captured
    assert "518396894" in captured
    assert "16.10" in captured
