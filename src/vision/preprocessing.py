"""Image preprocessing pipeline using OpenCV for Portuguese purchase receipts.

Handles downsampling, rotation correction, receipt boundary detection and perspective cropping,
fold/crease shadow removal via morphological background division, and thermal ink contrast
enhancement via CLAHE.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-04.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.types import ROTATION_CHOICES, RotationOption

if TYPE_CHECKING:
    from src.core.settings import Settings


@dataclass(frozen=True, slots=True)
class PreprocessOptions:
    """Configurable knobs for the preprocessing pipeline.

    Attributes:
        max_side: Maximum allowed dimension in pixels (width or height).
        rotation_deg: Clockwise rotation in degrees or 'auto' for aspect-ratio heuristic.
        auto_crop: Whether to attempt contour detection and perspective crop.
        remove_shadows: Whether to apply morphological background division.
        clahe_clip_limit: Contrast limit for CLAHE (0 disables CLAHE).
        clahe_tile_grid_size: Grid size for CLAHE adaptive histogram equalization.
        auto_orient_osd: Whether to probe Tesseract OSD to automatically detect and correct
            180-degree upside-down flips when rotation_deg is 'auto'.
        auto_deskew: Whether to apply continuous text-line deskew using OpenCV morphology
            and Hough transform. Defaults to False because adaptive 4-point cropping already
            straightens angled receipts while preserving unwarped pixels on upright receipts.
    """

    max_side: int = 1280
    rotation_deg: RotationOption = "auto"
    auto_crop: bool = True
    remove_shadows: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    auto_orient_osd: bool = True
    auto_deskew: bool = False
    tesseract_cmd: Path | None = None


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Outcome of preprocessing an image, including intermediate visualization steps.

    Attributes:
        image: Final processed BGR image ready for OCR inference.
        steps: Visual stages keyed by name ('bruta', 'rodada', 'recortada', 'filtrada').
        rotation_deg: Effective rotation applied in degrees (0, 90, 180, or 270).
    """

    image: np.ndarray
    steps: dict[str, np.ndarray]
    rotation_deg: int


def resize_max_side(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize image proportionally if its longest dimension exceeds max_side.

    Args:
        image: Source BGR image.
        max_side: Maximum allowed width or height in pixels.

    Returns:
        Resized or copied image.
    """
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy()

    scale = max_side / largest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


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


def order_corner_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) coordinates into [top-left, top-right, bottom-right, bottom-left].

    Uses clockwise angle sorting around the quadrilateral centroid so that strongly
    tilted rectangles never assign the same vertex to two corners.

    Args:
        pts: Array of shape (4, 2) containing point coordinates.

    Returns:
        Array of shape (4, 2) in float32 ordered [top-left, top-right, bottom-right, bottom-left].
    """
    pts_f = pts.reshape(4, 2).astype(np.float32)
    center = pts_f.mean(axis=0)

    angles = np.arctan2(pts_f[:, 1] - center[1], pts_f[:, 0] - center[0])
    cw_pts = pts_f[np.argsort(angles)]

    start_idx = int(np.argmin(cw_pts.sum(axis=1)))
    return np.roll(cw_pts, -start_idx, axis=0).astype(np.float32)


def _compute_paper_border_value(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, ...] | int:
    """Compute a neutral background fill color from paper pixels to avoid BORDER_REPLICATE streaks."""
    if image.ndim == 2:
        if mask is not None and np.any(mask > 0):
            return int(np.median(image[mask > 0]))
        return int(np.median(image))
    if mask is not None and np.any(mask > 0):
        med = np.median(image[mask > 0], axis=0)
    else:
        med = np.median(image.reshape(-1, image.shape[2]), axis=0)
    return tuple(int(round(float(x))) for x in med)


def _warp_perspective_quad(
    image: np.ndarray,
    corners: np.ndarray,
    border_value: tuple[int, ...] | int = (240, 240, 240),
) -> np.ndarray:
    """Warp 4 corner points into a straightened portrait BGR image via perspective transform.

    Args:
        image: Source BGR image.
        corners: Array of 4 (x, y) corner coordinates.
        border_value: Constant fill color for any out-of-bounds border pixels.

    Returns:
        Perspective-warped upright image with height >= width.
    """
    rect = order_corner_points(corners)
    tl, tr, br, bl = rect

    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    target_width = max(int(round(max(width_a, width_b))), 1)

    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    target_height = max(int(round(max(height_a, height_b))), 1)

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        transform_matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,  # type: ignore[arg-type]
    )

    warped_h, warped_w = warped.shape[:2]
    if warped_w > warped_h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


def _crop_receipt_with_status(
    image: np.ndarray,
    min_area_ratio: float = 0.20,
    min_tilt_for_perspective_deg: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """Adaptively segment receipt paper and either slice losslessly or warp perspective.

    Segments neutral bright paper (high HSV Value, low Saturation) covering the image center.
    When the receipt is nearly upright (tilt < min_tilt_for_perspective_deg), applies a lossless
    axis-aligned bounding-box crop so crumpled paper is neither sheared nor resampled. When
    genuinely angled (tilt >= min_tilt_for_perspective_deg), applies a padded 4-point perspective
    transform with constant paper-colored border fill.

    Returns:
        Tuple of (cropped_or_original_image, did_crop_boolean).
    """
    height, width = image.shape[:2]
    total_area = float(height * width)
    min_area = total_area * min_area_ratio
    max_area = total_area * 0.96

    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)
        paper_score = np.clip(v_chan - 1.2 * s_chan, 0, 255).astype(np.uint8)
    else:
        paper_score = image.copy()

    # Uniform image check (avoids false Otsu split on flat synthetic canvases)
    if int(paper_score.max()) - int(paper_score.min()) < 20:
        return image.copy(), False

    blurred = cv2.GaussianBlur(paper_score, (11, 11), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    paper_closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
    _, otsu_mask = cv2.threshold(paper_closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    center_pt = (width / 2.0, height / 2.0)
    min_center_dist = -0.10 * min(width, height)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        if area > max_area:
            continue

        hull = cv2.convexHull(contour)
        if cv2.pointPolygonTest(hull, center_pt, True) < min_center_dist:
            continue

        rot_rect = cv2.minAreaRect(hull)
        (cx, cy), (rect_w, rect_h), ang = rot_rect
        rect_area = rect_w * rect_h
        if rect_area < min_area or rect_area > max_area:
            continue

        solidity = area / max(rect_area, 1.0)
        if solidity < 0.65:
            continue

        abs_ang = abs(ang) % 90.0
        tilt_deg = min(abs_ang, 90.0 - abs_ang)

        if tilt_deg >= min_tilt_for_perspective_deg:
            # Tilted receipt: pad rotated box outward by 6% so edge characters are never cut
            padded_rect = ((cx, cy), (rect_w * 1.06, rect_h * 1.06), ang)
            box_pts = cv2.boxPoints(padded_rect)
            border_val = _compute_paper_border_value(image, otsu_mask)
            warped = _warp_perspective_quad(image, box_pts, border_value=border_val)
            return warped, True

        # Upright receipt: lossless axis-aligned bounding-box crop
        bx, by, bw, bh = cv2.boundingRect(hull)
        pad_x = int(round(bw * 0.015))
        pad_y = int(round(bh * 0.015))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(width, bx + bw + pad_x)
        y2 = min(height, by + bh + pad_y)
        cropped = image[y1:y2, x1:x2].copy()
        if cropped.shape[1] > cropped.shape[0]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        return cropped, True

    return image.copy(), False


def crop_receipt(image: np.ndarray, min_area_ratio: float = 0.20) -> np.ndarray:
    """Detect receipt boundary contour and warp perspective into an upright rectangle.

    Falls back cleanly to the input image when no distinct quadrilateral is found.

    Args:
        image: Source BGR image.
        min_area_ratio: Minimum ratio of receipt area relative to total image area.

    Returns:
        Warped rectangular receipt or uncropped image if detection fails.
    """
    warped, _ = _crop_receipt_with_status(image, min_area_ratio=min_area_ratio)
    return warped


def estimate_skew_angle(image: np.ndarray, max_skew_deg: float = 25.0) -> float:
    """Estimate dominant text-line skew angle in degrees using OpenCV morphology and Hough lines.

    Args:
        image: BGR or grayscale receipt image.
        max_skew_deg: Maximum absolute skew angle in degrees to consider.

    Returns:
        Estimated skew angle in degrees in [-max_skew_deg, max_skew_deg], or 0.0 when no text
        lines are detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0

    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    if int(blackhat.max()) < 15:
        return 0.0

    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    strip_w = max(15, w // 25)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (strip_w, 1))
    connected = cv2.dilate(binary, connect_kernel, iterations=1)

    lines = cv2.HoughLinesP(
        connected,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=max(30, w // 8),
        maxLineGap=max(10, w // 30),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if np.hypot(dx, dy) < 1.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if -max_skew_deg <= angle <= max_skew_deg:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def deskew_image(
    image: np.ndarray,
    min_angle_deg: float = 0.5,
    max_skew_deg: float = 25.0,
) -> tuple[np.ndarray, float]:
    """Deskew receipt image continuously using OpenCV affine rotation around text baselines.

    Args:
        image: Source image (BGR or grayscale).
        min_angle_deg: Minimum absolute angle required to trigger resampling.
        max_skew_deg: Maximum allowed skew angle in degrees.

    Returns:
        Tuple of (deskewed_image, applied_angle_deg).
    """
    angle = estimate_skew_angle(image, max_skew_deg=max_skew_deg)
    if abs(angle) < min_angle_deg:
        return image.copy(), 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_a = abs(float(rot_mat[0, 0]))
    sin_a = abs(float(rot_mat[0, 1]))
    new_w = max(1, int(round((h * sin_a) + (w * cos_a))))
    new_h = max(1, int(round((h * cos_a) + (w * sin_a))))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    border_val = _compute_paper_border_value(image)
    deskewed = cv2.warpAffine(
        image,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,  # type: ignore[arg-type]
    )
    return deskewed, angle


def _detect_qr_vertical_ratio(image: np.ndarray) -> float | None:
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


def detect_upside_down_qr(image: np.ndarray) -> bool:
    """Detect 180-degree upside-down orientation via OpenCV QR code footer location.

    On Portuguese fiscal receipts, the AT QR code is printed in the footer (bottom half).
    If a QR code is detected in the upper 25% of a portrait image, the receipt is upside-down.

    Args:
        image: Portrait BGR or grayscale receipt image.

    Returns:
        True when a QR code is detected in the top 25% of the image, False otherwise.
    """
    ratio = _detect_qr_vertical_ratio(image)
    return ratio is not None and ratio < 0.25


def attenuate_shadows(gray: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """Normalize background lighting on crumpled paper via morphological division.

    Args:
        gray: Single-channel grayscale image.
        kernel_size: Size of the square morphological structuring element.

    Returns:
        Shadow-attenuated grayscale image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blur_ksize = kernel_size | 1
    background = cv2.GaussianBlur(background, (blur_ksize, blur_ksize), 0)
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def enhance_contrast_clahe(
    gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to recover thermal print.

    Args:
        gray: Single-channel grayscale image.
        clip_limit: Threshold for contrast limiting. Values <= 0 return input untouched.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced grayscale image.
    """
    if clip_limit <= 0:
        return gray.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


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


def preprocess(image: np.ndarray, opts: PreprocessOptions | None = None) -> PreprocessResult:
    """Execute the full preprocessing pipeline on a receipt image.

    Args:
        image: Input image (BGR or Grayscale).
        opts: Processing options. When None, standard defaults are applied.

    Returns:
        PreprocessResult containing the OCR-ready image and all intermediate steps.

    Raises:
        ValueError: If the input image is empty or has zero area.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must not be empty or zero-sized.")

    options = opts or PreprocessOptions()

    # Normalise to 3-channel BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    # Step 1: Resize
    raw_resized = resize_max_side(bgr, max_side=options.max_side)
    steps: dict[str, np.ndarray] = {"bruta": raw_resized.copy()}
    crop_source = (
        resize_max_side(bgr, max_side=max(options.max_side * 2, options.max_side)) if options.auto_crop else raw_resized
    )

    # Step 2 & 3: Orientation, 4-point perspective crop, and continuous deskew
    if options.rotation_deg != "auto":
        oriented, effective_rotation = orient_image(raw_resized, rotation_deg=options.rotation_deg)
        if options.auto_crop:
            oriented_source, _ = orient_image(crop_source, rotation_deg=options.rotation_deg)
            cropped = resize_max_side(crop_receipt(oriented_source), max_side=options.max_side)
        else:
            cropped = oriented.copy()
        if options.auto_deskew:
            cropped, _ = deskew_image(cropped)
        steps["rodada"] = oriented.copy()
        steps["recortada"] = cropped.copy()
    else:
        if options.auto_crop:
            cropped_hires, did_crop = _crop_receipt_with_status(crop_source)
            if did_crop:
                cropped = resize_max_side(cropped_hires, max_side=options.max_side)
                oriented_full, effective_rotation = orient_image(raw_resized, rotation_deg="auto")
                if options.auto_deskew:
                    cropped, _ = deskew_image(cropped)
                steps["rodada"] = oriented_full.copy()
                steps["recortada"] = cropped.copy()
            else:
                oriented, effective_rotation = orient_image(raw_resized, rotation_deg="auto")
                oriented_source, _ = orient_image(crop_source, rotation_deg="auto")
                cropped = resize_max_side(crop_receipt(oriented_source), max_side=options.max_side)
                if options.auto_deskew:
                    cropped, _ = deskew_image(cropped)
                steps["rodada"] = oriented.copy()
                steps["recortada"] = cropped.copy()
        else:
            oriented, effective_rotation = orient_image(raw_resized, rotation_deg="auto")
            cropped = oriented.copy()
            if options.auto_deskew:
                cropped, _ = deskew_image(cropped)
            steps["rodada"] = oriented.copy()
            steps["recortada"] = cropped.copy()

    # Step 4: Shadow attenuation and CLAHE contrast enhancement
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    if options.remove_shadows:
        gray = attenuate_shadows(gray)

    if options.clahe_clip_limit > 0:
        gray = enhance_contrast_clahe(
            gray, clip_limit=options.clahe_clip_limit, tile_grid_size=options.clahe_tile_grid_size
        )

    filtered_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Step 5: Semantic flip verification via QR position ratio and Tesseract OSD
    if options.rotation_deg == "auto" and options.auto_orient_osd:
        qr_ratio = _detect_qr_vertical_ratio(cropped)
        if qr_ratio is None:
            qr_ratio = _detect_qr_vertical_ratio(filtered_bgr)

        # In Portugal, QR codes are never printed in the header (top 25%).
        # If detected at ratio < 0.25, the receipt is upside down.
        # If detected at ratio >= 0.30, it is confirmed right-side up.
        is_upside_down_qr = (qr_ratio is not None) and (qr_ratio < 0.25)
        is_confirmed_upright_qr = (qr_ratio is not None) and (qr_ratio >= 0.30)

        osd_rotate = detect_orientation_osd(cropped, min_confidence=2.5, tesseract_cmd=options.tesseract_cmd)
        if osd_rotate == 0:
            osd_rotate = detect_orientation_osd(filtered_bgr, min_confidence=2.5, tesseract_cmd=options.tesseract_cmd)

        if not is_confirmed_upright_qr and (is_upside_down_qr or osd_rotate == 180):
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_180)
            steps["recortada"] = cv2.rotate(steps["recortada"], cv2.ROTATE_180)
            effective_rotation = (effective_rotation + 180) % 360
        elif not is_confirmed_upright_qr and osd_rotate == 90:
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_90_CLOCKWISE)
            steps["recortada"] = cv2.rotate(steps["recortada"], cv2.ROTATE_90_CLOCKWISE)
            effective_rotation = (effective_rotation + 90) % 360
        elif not is_confirmed_upright_qr and osd_rotate == 270:
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            steps["recortada"] = cv2.rotate(steps["recortada"], cv2.ROTATE_90_COUNTERCLOCKWISE)
            effective_rotation = (effective_rotation + 270) % 360

    steps["filtrada"] = filtered_bgr.copy()

    return PreprocessResult(
        image=filtered_bgr,
        steps=steps,
        rotation_deg=effective_rotation,
    )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Run preprocessing on a receipt and display before/after windows.

    Supports CLI flags for targeting specific receipt files, testing rotation overrides,
    and toggling crop, shadow attenuation, or automatic OSD flip.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from PIL import Image

    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings

    parser = argparse.ArgumentParser(description="Receipt image preprocessing preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Specific receipt image path")
    parser.add_argument(
        "--rotate",
        "-r",
        choices=ROTATION_CHOICES,
        default="auto",
        help="Rotation angle in degrees or 'auto'",
    )
    parser.add_argument("--no-crop", action="store_true", help="Disable receipt boundary cropping")
    parser.add_argument("--no-shadows", action="store_true", help="Disable crease shadow attenuation")
    parser.add_argument("--no-osd", action="store_true", help="Disable automatic Tesseract OSD flip")
    parser.add_argument("--deskew", action="store_true", help="Enable continuous text-line deskew")
    parser.add_argument("--no-deskew", action="store_true", help="Disable continuous text-line deskew")
    parser.add_argument("--save", "-s", type=str, default=None, help="Save side-by-side comparison image to path")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    loaded = resolve_and_load_receipt(args.file, settings)
    if loaded is None:
        return
    target_path, raw_image = loaded

    rot_opt: RotationOption = "auto" if args.rotate == "auto" else int(args.rotate)  # type: ignore[assignment]
    opts = PreprocessOptions(
        rotation_deg=rot_opt,
        auto_crop=not args.no_crop,
        remove_shadows=not args.no_shadows,
        auto_orient_osd=not args.no_osd,
        auto_deskew=bool(args.deskew and not args.no_deskew),
    )

    result = preprocess(raw_image, opts=opts)

    # Prepare side-by-side comparison (Before on left, After on right)
    preview_height = 1000
    raw_scaled = resize_max_side(raw_image, max_side=1280)
    proc_img = result.image

    w_raw = max(1, int(round(raw_scaled.shape[1] * (preview_height / raw_scaled.shape[0]))))
    w_proc = max(1, int(round(proc_img.shape[1] * (preview_height / proc_img.shape[0]))))

    side_raw = cv2.resize(raw_scaled, (w_raw, preview_height), interpolation=cv2.INTER_AREA)
    side_proc = cv2.resize(proc_img, (w_proc, preview_height), interpolation=cv2.INTER_AREA)
    separator = np.full((preview_height, 12, 3), 40, dtype=np.uint8)

    comparison = np.hstack([side_raw, separator, side_proc])

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), comparison)
        print(f"Saved side-by-side comparison to: {save_path}")
    else:
        title = f"Antes vs Depois ({target_path.name})"
        Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)).show(title=title)


if __name__ == "__main__":
    main()
