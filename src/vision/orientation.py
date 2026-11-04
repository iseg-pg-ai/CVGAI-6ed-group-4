"""Orientation analysis, QR-based flip detection, and Tesseract OSD probes.

Ref: Docs/implementation_plan.md, section 7, WP-04.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.core.types import RotationOption


def orient_image(image: np.ndarray, rotation_deg: RotationOption = "auto") -> tuple[np.ndarray, int]:
    """Rotate image by explicit degrees or detect sideways smartphone captures.

    Portuguese receipts are predominantly vertical ribbons (height >= width). Under 'auto',
    any landscape photograph (width > height) is rotated 90 degrees clockwise.

    Args:
        image: Source image.
        rotation_deg: Rotation in degrees (0, 90, 180, 270) or 'auto'.

    Returns:
        Tuple of (rotated_image, effective_rotation_degrees).
    """
    if rotation_deg == 0:
        return image.copy(), 0
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180), 180
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 270

    height, width = image.shape[:2]
    if width > height:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90
    return image.copy(), 0


def detect_qr_vertical_ratio(image: np.ndarray) -> float | None:
    """Return vertical centroid ratio (0.0 to 1.0) of QR code if detected, else None."""
    h, w = image.shape[:2]
    if h <= w:
        return None
    detector = cv2.QRCodeDetector()
    try:
        found, points = detector.detect(image)
    except cv2.error:
        return None
    if not found or points is None or points.size == 0:
        return None
    pts = points.reshape(-1, 2)
    return float(pts[:, 1].mean()) / float(h)


# Backward-compatible alias
_detect_qr_vertical_ratio = detect_qr_vertical_ratio


def detect_upside_down_qr(image: np.ndarray) -> bool:
    """Detect 180-degree upside-down orientation via OpenCV QR code footer location.

    On Portuguese fiscal receipts, the AT QR code is printed in the footer (bottom half).
    If a QR code is detected in the upper 25% of a portrait image, the receipt is upside-down.

    Args:
        image: Portrait BGR or grayscale receipt image.

    Returns:
        True when a QR code is detected in the top 25% of the image, False otherwise.
    """
    ratio = detect_qr_vertical_ratio(image)
    return ratio is not None and ratio < 0.25


def detect_orientation_osd(
    image: np.ndarray,
    tesseract_cmd: Path | str | None = None,
    min_confidence: float = 2.0,
) -> int:
    """Detect orientation angle in degrees (0, 90, 180, 270) using Tesseract OSD.

    Runs a lightweight OSD (Orientation and Script Detection) probe. Returns 0 if Tesseract
    is unavailable, if recognition fails, or if orientation confidence is below min_confidence.
    Never raises exceptions.

    Args:
        image: Grayscale or BGR image to inspect.
        tesseract_cmd: Optional explicit path to the tesseract binary.
        min_confidence: Minimum orientation confidence threshold required to accept rotation.

    Returns:
        Degrees to rotate clockwise (0, 90, 180, or 270).
    """
    cmd = str(tesseract_cmd) if tesseract_cmd else shutil.which("tesseract")
    if not cmd and os.name == "nt":
        default_win = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if default_win.is_file():
            cmd = str(default_win)

    if not cmd:
        return 0

    success, buf = cv2.imencode(".png", image)
    if not success:
        return 0

    try:
        proc = subprocess.run(
            [cmd, "stdin", "stdout", "--psm", "0"],
            input=buf.tobytes(),
            capture_output=True,
            timeout=5.0,
        )
        if proc.returncode != 0:
            return 0
        output = proc.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return 0

    rotate_deg = 0
    confidence = 0.0
    for line in output.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("Rotate:"):
            with contextlib.suppress(ValueError):
                rotate_deg = int(line_clean.split(":", 1)[1].strip())
        elif line_clean.startswith("Orientation confidence:"):
            with contextlib.suppress(ValueError):
                confidence = float(line_clean.split(":", 1)[1].strip())

    if confidence >= min_confidence and rotate_deg in {90, 180, 270}:
        return rotate_deg
    return 0


def resolve_semantic_flip(
    cropped: np.ndarray,
    filtered_bgr: np.ndarray,
    tesseract_cmd: Path | str | None = None,
) -> int:
    """Determine whether an image needs a 180 or 90/270 flip via QR location and Tesseract OSD.

    In Portugal, fiscal AT QR codes are printed in the footer. If detected at ratio < 0.25,
    the receipt is upside-down. If detected at ratio >= 0.30, it is confirmed upright.
    OSD probe is used as a secondary check when QR position does not confirm orientation.

    Args:
        cropped: Cropped receipt image.
        filtered_bgr: Shadow-attenuated and contrast-enhanced image.
        tesseract_cmd: Optional path to tesseract binary.

    Returns:
        Clockwise degrees to rotate (0, 90, 180, or 270).
    """
    qr_ratio = detect_qr_vertical_ratio(cropped)
    if qr_ratio is None:
        qr_ratio = detect_qr_vertical_ratio(filtered_bgr)

    is_upside_down_qr = (qr_ratio is not None) and (qr_ratio < 0.25)
    is_confirmed_upright_qr = (qr_ratio is not None) and (qr_ratio >= 0.30)

    osd_rotate = detect_orientation_osd(cropped, min_confidence=2.5, tesseract_cmd=tesseract_cmd)
    if osd_rotate == 0:
        osd_rotate = detect_orientation_osd(filtered_bgr, min_confidence=2.5, tesseract_cmd=tesseract_cmd)

    if not is_confirmed_upright_qr:
        if is_upside_down_qr or osd_rotate == 180:
            return 180
        if osd_rotate in (90, 270):
            return osd_rotate
    return 0
