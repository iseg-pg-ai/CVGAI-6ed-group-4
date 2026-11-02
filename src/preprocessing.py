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
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from src.types import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from src.settings import Settings

RotationOption = Literal["auto", 0, 90, 180, 270]


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
    """

    max_side: int = 1280
    rotation_deg: RotationOption = "auto"
    auto_crop: bool = True
    remove_shadows: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    auto_orient_osd: bool = True


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

    Args:
        pts: Array of shape (4, 2) containing point coordinates.

    Returns:
        Array of shape (4, 2) ordered consistently for perspective transform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest x + y; bottom-right has largest x + y
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest y - x; bottom-left has largest y - x
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def crop_receipt(image: np.ndarray, min_area_ratio: float = 0.20) -> np.ndarray:
    """Detect receipt boundary contour and warp perspective into an upright rectangle.

    Falls back cleanly to the input image when no distinct quadrilateral is found.

    Args:
        image: Source BGR image.
        min_area_ratio: Minimum ratio of receipt area relative to total image area.

    Returns:
        Warped rectangular receipt or uncropped image if detection fails.
    """
    height, width = image.shape[:2]
    total_area = height * width
    min_area = total_area * min_area_ratio

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to bridge faint thermal paper boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy()

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = order_corner_points(pts)
            tl, tr, br, bl = rect

            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            target_width = max(int(round(max(width_a, width_b))), 1)

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
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
            warped = cv2.warpPerspective(image, transform_matrix, (target_width, target_height), flags=cv2.INTER_CUBIC)

            # Ensure portrait orientation for receipts
            warped_h, warped_w = warped.shape[:2]
            if warped_w > warped_h:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            return warped

    return image.copy()


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
        default_win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
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

    # Step 2: Orientation (Stage 1: Aspect ratio heuristic)
    oriented, effective_rotation = orient_image(raw_resized, rotation_deg=options.rotation_deg)
    steps["rodada"] = oriented.copy()

    # Step 3: Cropping
    cropped = crop_receipt(oriented) if options.auto_crop else oriented.copy()
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

    # Step 5: Semantic flip verification via Tesseract OSD (Stage 2)
    if options.rotation_deg == "auto" and options.auto_orient_osd:
        osd_rotate = detect_orientation_osd(filtered_bgr)
        if osd_rotate == 180:
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_180)
            effective_rotation = (effective_rotation + 180) % 360
        elif osd_rotate == 90:
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_90_CLOCKWISE)
            effective_rotation = (effective_rotation + 90) % 360
        elif osd_rotate == 270:
            filtered_bgr = cv2.rotate(filtered_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

    from src.settings import get_settings

    parser = argparse.ArgumentParser(description="Receipt image preprocessing preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Specific receipt image path")
    parser.add_argument(
        "--rotate",
        "-r",
        choices=["auto", "0", "90", "180", "270"],
        default="auto",
        help="Rotation angle in degrees or 'auto'",
    )
    parser.add_argument("--no-crop", action="store_true", help="Disable receipt boundary cropping")
    parser.add_argument("--no-shadows", action="store_true", help="Disable crease shadow attenuation")
    parser.add_argument("--no-osd", action="store_true", help="Disable automatic Tesseract OSD flip")
    parser.add_argument("--save", "-s", type=str, default=None, help="Save side-by-side comparison image to path")

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
            [p for p in settings.receipts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
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

    rot_opt: RotationOption = "auto" if args.rotate == "auto" else int(args.rotate)  # type: ignore[assignment]
    opts = PreprocessOptions(
        rotation_deg=rot_opt,
        auto_crop=not args.no_crop,
        remove_shadows=not args.no_shadows,
        auto_orient_osd=not args.no_osd,
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
