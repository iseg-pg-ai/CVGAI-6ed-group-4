"""Portuguese Tax Authority (AT) QR code reader and payload parser.

Extracts and parses the Portaria n.º 195/2020 key-value QR payload (* separated pairs) from
receipt images using cv2.QRCodeDetector with fallback enhancements. Used for cross-validation
and comparison against OCR/LLM extractions.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-07.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.settings import Settings


def parse_at_qr_payload(payload: str) -> dict[str, str] | None:
    """Parse a Portuguese Tax Authority (AT) QR code payload into structured fields.

    The official AT format uses '*' separated KEY:VALUE pairs (e.g. A:500100144*B:999999990*...).
    Mandatory minimum keys include 'A' (merchant NIF) or 'O' (document total).

    Args:
        payload: Decoded QR code string.

    Returns:
        Dictionary of raw AT keys and canonical convenience aliases, or None if the payload
        does not conform to the AT invoice QR specification.
    """
    if not payload or "*" not in payload:
        return None

    tokens = payload.split("*")
    data: dict[str, str] = {}

    for token in tokens:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        key_clean = key.strip()
        value_clean = value.strip()
        if key_clean:
            data[key_clean] = value_clean

    # Validate that at least merchant NIF ('A') or total ('O') is present
    if "A" not in data and "O" not in data:
        return None

    # Populate canonical convenience aliases for pipeline cross-validation
    if "A" in data:
        data["nif_comerciante"] = data["A"]

    if "B" in data:
        client_nif = data["B"]
        # In Portugal, '999999990' denotes Final Consumer (no customer taxpayer number)
        if client_nif == "999999990" or not client_nif:
            data["nif_cliente"] = ""
        else:
            data["nif_cliente"] = client_nif

    if "F" in data:
        raw_date = data["F"]
        # Format AAAAMMDD -> YYYY-MM-DD
        if len(raw_date) == 8 and raw_date.isdigit():
            data["data"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        else:
            data["data"] = raw_date

    if "O" in data:
        data["total"] = data["O"]

    if "H" in data:
        data["atcud"] = data["H"]

    if "D" in data:
        data["tipo_doc"] = data["D"]

    return data


def _decode_single_image(img: np.ndarray, qr_detector: Any) -> str | None:
    """Attempt decoding using pyzbar first, falling back to cv2.QRCodeDetector."""
    # Attempt with pyzbar if installed
    with contextlib.suppress(Exception):
        import pyzbar.pyzbar as pyzbar  # type: ignore[import-not-found,import-untyped]

        results = pyzbar.decode(img)
        for item in results:
            text = item.data.decode("utf-8", errors="ignore").strip()
            if text:
                return text

    # Attempt with OpenCV QRCodeDetector
    with contextlib.suppress(Exception):
        decoded_text, _, _ = qr_detector.detectAndDecode(img)
        text_clean = str(decoded_text).strip()
        if text_clean:
            return text_clean

    return None


def _detect_and_heal_qr_regions(gray: np.ndarray, qr_detector: Any) -> str | None:
    """Locate candidate square QR blobs and heal thermal printer dead-pin streaks.

    Thermal printers frequently have burnt or dusty heating pins that slice thin vertical
    white streaks through the QR code. This method fuses modules into candidate blobs,
    crops them, and applies horizontal morphological erosion to bridge the dead-pin gaps.
    """
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blob = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh)
        area = bw * bh
        if 0.70 <= aspect <= 1.40 and 10000 <= area <= 700000:
            candidates.append((x, y, bw, bh))

    for x, y, bw, bh in candidates:
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + bw + pad)
        y2 = min(height, y + bh + pad)
        crop_gray = gray[y1:y2, x1:x2]

        # 1. Direct decode on crop
        decoded = _decode_single_image(crop_gray, qr_detector)
        if decoded:
            return decoded

        # 2. Local Otsu threshold + horizontal streak healing
        _, bin_crop = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for kw in (1, 2, 3, 4):
            heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
            healed = cv2.erode(bin_crop, heal_kernel)
            decoded = _decode_single_image(healed, qr_detector)
            if decoded:
                return decoded

    return None


def read(
    image: np.ndarray,
    raw_image: np.ndarray | None = None,
    detector: Any | None = None,
) -> dict[str, str] | None:
    """Detect and decode an AT QR code from a receipt image.

    Applies multi-stage decoding resilience: direct detection, grayscale detection,
    adaptive thresholding, candidate blob localization with thermal printhead streak healing,
    and high-resolution raw image fallback.

    Args:
        image: BGR or Grayscale input image (e.g. preprocessed).
        raw_image: Optional unscaled raw image for high-resolution QR fallback.
        detector: Optional custom detector instance for testing or dependency injection.

    Returns:
        Parsed AT QR dictionary or None if no valid AT QR code is found.

    Raises:
        ValueError: If the input image is empty or has zero area.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must not be empty or zero-sized.")

    qr_detector = detector or cv2.QRCodeDetector()

    # Stage 1: Try direct detection on input image
    decoded = _decode_single_image(image, qr_detector)
    if decoded:
        parsed = parse_at_qr_payload(decoded)
        if parsed:
            return parsed

    # Stage 2: Try grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    decoded = _decode_single_image(gray, qr_detector)
    if decoded:
        parsed = parse_at_qr_payload(decoded)
        if parsed:
            return parsed

    # Stage 3: Try adaptive Gaussian thresholding (recovers faint thermal print)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    decoded = _decode_single_image(thresh, qr_detector)
    if decoded:
        parsed = parse_at_qr_payload(decoded)
        if parsed:
            return parsed

    # Stage 4: Try localized candidate blob search with thermal streak healing
    decoded = _detect_and_heal_qr_regions(gray, qr_detector)
    if decoded:
        parsed = parse_at_qr_payload(decoded)
        if parsed:
            return parsed

    # Stage 5: Try unscaled raw image if available (preserves tiny pixel modules)
    if raw_image is not None and raw_image is not image:
        raw_result = read(raw_image, detector=detector)
        if raw_result:
            return raw_result

    return None


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to inspect AT QR code extraction on a receipt.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.preprocessing import PreprocessOptions, preprocess
    from src.settings import get_settings

    parser = argparse.ArgumentParser(description="Portuguese AT QR Code reader preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    target_path: Path | None = None
    if args.file:
        candidate = Path(args.file)
        if not candidate.is_file():
            print(f"Error: Specified file not found: {candidate}")
            return
        target_path = candidate
    else:
        if not settings.receipts_dir.is_dir():
            print(f"Receipts directory not found: {settings.receipts_dir}")
            return
        receipt_files = sorted(
            [
                p
                for p in settings.receipts_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not receipt_files:
            print(f"No receipt images found in {settings.receipts_dir}")
            return
        target_path = receipt_files[0]

    print(f"Loading receipt: {target_path.name}")
    raw_image = cv2.imread(str(target_path))
    if raw_image is None:
        print(f"Error: Unable to read image file: {target_path}")
        return

    if args.no_preprocess:
        image_to_read = raw_image
    else:
        print("Preprocessing receipt...")
        prep_result = preprocess(raw_image, PreprocessOptions())
        image_to_read = prep_result.image

    print("Detecting and decoding AT QR code...")
    result = read(image_to_read, raw_image=raw_image)

    if not result:
        print("No AT QR code detected in this document (e.g. payment slip or damaged QR).")
        return

    print("\n--- AT Tax Authority QR Data Found ---")
    for key, val in result.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
