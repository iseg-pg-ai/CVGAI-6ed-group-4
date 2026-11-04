"""Portuguese Tax Authority (AT) QR code reader and payload parser.

Extracts and parses the Portaria n.º 195/2020 key-value QR payload (* separated pairs) from
receipt images using cv2.QRCodeDetector with fallback enhancements. Used for cross-validation
and comparison against OCR/LLM extractions.

Ref: Docs/implementation_plan.md, section 7, WP-07.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.core.types import CONSUMIDOR_FINAL_NIF, validate_image
from src.vision.geometry import compute_paper_border_value

if TYPE_CHECKING:
    from src.core.settings import Settings


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
        if client_nif == CONSUMIDOR_FINAL_NIF or not client_nif:
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


def _decode_all_codes(img: np.ndarray, qr_detector: Any) -> list[str]:
    """Attempt decoding all candidate codes using pyzbar and cv2.QRCodeDetector."""
    results: list[str] = []
    # Attempt with pyzbar if installed
    with contextlib.suppress(Exception):
        import pyzbar.pyzbar as pyzbar  # type: ignore[import-not-found,import-untyped]

        for item in pyzbar.decode(img):
            text = item.data.decode("utf-8", errors="ignore").strip()
            if text and text not in results:
                results.append(text)

    # Attempt with OpenCV QRCodeDetector
    with contextlib.suppress(Exception):
        if hasattr(qr_detector, "detectAndDecodeMulti"):
            retval, decoded_info, _, _ = qr_detector.detectAndDecodeMulti(img)
            if retval and decoded_info is not None:
                for item in decoded_info:
                    clean = str(item).strip()
                    if clean and clean not in results:
                        results.append(clean)
        else:
            decoded_text, _, _ = qr_detector.detectAndDecode(img)
            text_clean = str(decoded_text).strip()
            if text_clean and text_clean not in results:
                results.append(text_clean)

    return results


def _decode_first_at_qr(img: np.ndarray, qr_detector: Any) -> dict[str, str] | None:
    """Return the first valid AT QR payload decoded from img, or None."""
    for text in _decode_all_codes(img, qr_detector):
        parsed = parse_at_qr_payload(text)
        if parsed is not None:
            return parsed
    return None


def _detect_and_heal_qr_regions(gray: np.ndarray, qr_detector: Any) -> dict[str, str] | None:
    """Locate candidate square QR blobs and heal thermal printer dead-pin streaks.

    Thermal printers frequently have burnt or dusty heating pins that slice thin vertical
    white streaks through the QR code. This method fuses modules into candidate blobs,
    crops them, and applies horizontal morphological erosion to bridge the dead-pin gaps.
    """
    height, width = gray.shape[:2]
    total_area = height * width
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
        if 0.65 <= aspect <= 1.45 and (0.005 * total_area) <= area <= (0.75 * total_area):
            candidates.append((x, y, bw, bh))

    for x, y, bw, bh in candidates:
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + bw + pad)
        y2 = min(height, y + bh + pad)
        crop_gray = gray[y1:y2, x1:x2]

        # 1. Direct decode on crop
        parsed = _decode_first_at_qr(crop_gray, qr_detector)
        if parsed is not None:
            return parsed

        # 2. Local Otsu threshold + horizontal streak healing
        _, bin_crop = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for kw in (1, 2, 3, 4):
            heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
            healed = cv2.erode(bin_crop, heal_kernel)
            parsed = _decode_first_at_qr(healed, qr_detector)
            if parsed is not None:
                return parsed

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
    validate_image(image)

    qr_detector = detector or cv2.QRCodeDetector()

    # Stage 1: Try direct detection on input image
    parsed = _decode_first_at_qr(image, qr_detector)
    if parsed is not None:
        return parsed

    # Stage 2: Try grayscale if input was 3-channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if image.ndim == 3:
        parsed = _decode_first_at_qr(gray, qr_detector)
        if parsed is not None:
            return parsed

    # Stage 3: Try adaptive Gaussian thresholding (recovers faint thermal print)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    parsed = _decode_first_at_qr(thresh, qr_detector)
    if parsed is not None:
        return parsed

    # Stage 4: Try localized candidate blob search with thermal streak healing
    blob_result = _detect_and_heal_qr_regions(gray, qr_detector)
    if blob_result is not None:
        return blob_result

    # Stage 5: Try unscaled raw image if available (preserves tiny pixel modules)
    if raw_image is not None and raw_image is not image:
        raw_result = read(raw_image, detector=detector)
        if raw_result is not None:
            return raw_result

    return None


def find_qr_bounding_boxes(image: np.ndarray, detector: Any | None = None) -> list[tuple[int, int, int, int]]:
    """Locate bounding boxes (x1, y1, x2, y2) of any QR codes present in the image.

    Args:
        image: Source BGR or grayscale image.
        detector: Optional injected detector instance.

    Returns:
        List of (x1, y1, x2, y2) boxes clamped to image boundaries.
    """
    validate_image(image)
    h, w = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []

    # 1. Try PyZbar polygon rectangles
    with contextlib.suppress(Exception):
        import pyzbar.pyzbar as pyzbar  # type: ignore[import-not-found,import-untyped]

        codes = pyzbar.decode(image)
        for code in codes:
            if code.type == "QRCODE":
                rect = code.rect
                x1 = max(0, rect.left)
                y1 = max(0, rect.top)
                x2 = min(w, rect.left + rect.width)
                y2 = min(h, rect.top + rect.height)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

    # 2. Try OpenCV QRCodeDetector multi-detection points
    qr_detector = detector or cv2.QRCodeDetector()
    with contextlib.suppress(Exception):
        if hasattr(qr_detector, "detectMulti"):
            found, points = qr_detector.detectMulti(image)
            if found and points is not None:
                for pts in points:
                    pts_reshaped = pts.reshape(-1, 2)
                    x1 = max(0, int(np.floor(pts_reshaped[:, 0].min())))
                    y1 = max(0, int(np.floor(pts_reshaped[:, 1].min())))
                    x2 = min(w, int(np.ceil(pts_reshaped[:, 0].max())))
                    y2 = min(h, int(np.ceil(pts_reshaped[:, 1].max())))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
        else:
            found, points = qr_detector.detect(image)
            if found and points is not None and points.size >= 8:
                pts_reshaped = points.reshape(-1, 2)
                x1 = max(0, int(np.floor(pts_reshaped[:, 0].min())))
                y1 = max(0, int(np.floor(pts_reshaped[:, 1].min())))
                x2 = min(w, int(np.ceil(pts_reshaped[:, 0].max())))
                y2 = min(h, int(np.ceil(pts_reshaped[:, 1].max())))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

    # Remove duplicates or near-identical boxes
    unique_boxes: list[tuple[int, int, int, int]] = []
    for b in boxes:
        is_dup = False
        for ub in unique_boxes:
            if abs(b[0] - ub[0]) < 10 and abs(b[1] - ub[1]) < 10 and abs(b[2] - ub[2]) < 10 and abs(b[3] - ub[3]) < 10:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append(b)

    return unique_boxes


def mask_qr_region(image: np.ndarray, padding: int = 12, detector: Any | None = None) -> np.ndarray:
    """Return a copy of the image with detected QR code areas overwritten by paper background.

    Args:
        image: Source BGR or grayscale image.
        padding: Pixel expansion applied to all four sides of each detected box.
        detector: Optional injected detector instance.

    Returns:
        New image array with QR regions masked out, or an exact copy if no QR is found.
    """
    validate_image(image)
    boxes = find_qr_bounding_boxes(image, detector=detector)
    if not boxes:
        return image.copy()

    h, w = image.shape[:2]
    masked = image.copy()
    paper_fill = compute_paper_border_value(image)

    for x1, y1, x2, y2 in boxes:
        pad_x1 = max(0, x1 - padding)
        pad_y1 = max(0, y1 - padding)
        pad_x2 = min(w, x2 + padding)
        pad_y2 = min(h, y2 + padding)

        if masked.ndim == 2:
            fill_val = paper_fill if isinstance(paper_fill, int) else int(paper_fill[0])
            masked[pad_y1:pad_y2, pad_x1:pad_x2] = fill_val
        else:
            fill_tuple = paper_fill if isinstance(paper_fill, tuple) else (paper_fill, paper_fill, paper_fill)
            masked[pad_y1:pad_y2, pad_x1:pad_x2] = fill_tuple

    return masked


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to inspect AT QR code extraction on a receipt.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="Portuguese AT QR Code reader preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    loaded = resolve_and_load_receipt(args.file, settings)
    if loaded is None:
        return
    target_path, raw_image = loaded

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
