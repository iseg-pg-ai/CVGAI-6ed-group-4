"""Image preprocessing pipeline using OpenCV for Portuguese purchase receipts.

Handles downsampling, rotation correction, receipt boundary detection and perspective cropping,
fold/crease shadow removal via morphological background division, and thermal ink contrast
enhancement via CLAHE.

Ref: Docs/implementation_plan.md, section 7, WP-04.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.types import ROTATION_CHOICES, RotationOption, validate_image
from src.vision.geometry import (
    _compute_paper_border_value,
    _crop_receipt_with_status,
    _warp_perspective_quad,
    compute_paper_border_value,
    crop_receipt,
    crop_receipt_with_status,
    order_corner_points,
    resize_max_side,
    warp_perspective_quad,
)
from src.vision.orientation import (
    _detect_qr_vertical_ratio,
    detect_orientation_osd,
    detect_qr_vertical_ratio,
    detect_upside_down_qr,
    orient_image,
    resolve_semantic_flip,
)
from src.vision.photometry import (
    attenuate_shadows,
    deskew_image,
    enhance_contrast_clahe,
    estimate_skew_angle,
)

if TYPE_CHECKING:
    from src.core.settings import Settings

__all__ = [
    "PreprocessOptions",
    "PreprocessResult",
    "_compute_paper_border_value",
    "_crop_receipt_with_status",
    "_detect_qr_vertical_ratio",
    "_warp_perspective_quad",
    "attenuate_shadows",
    "compute_paper_border_value",
    "crop_receipt",
    "crop_receipt_with_status",
    "deskew_image",
    "detect_orientation_osd",
    "detect_qr_vertical_ratio",
    "detect_upside_down_qr",
    "enhance_contrast_clahe",
    "estimate_skew_angle",
    "main",
    "order_corner_points",
    "orient_image",
    "preprocess",
    "resize_max_side",
    "warp_perspective_quad",
]


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
        tesseract_cmd: Optional explicit path to the tesseract executable.
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
    validate_image(image)

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
    elif options.auto_crop:
        cropped_hires, did_crop = _crop_receipt_with_status(crop_source)
        oriented, effective_rotation = orient_image(raw_resized, rotation_deg="auto")
        if did_crop:
            cropped = resize_max_side(cropped_hires, max_side=options.max_side)
        else:
            oriented_source, _ = orient_image(crop_source, rotation_deg="auto")
            cropped = resize_max_side(crop_receipt(oriented_source), max_side=options.max_side)
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
        target_rotate = resolve_semantic_flip(cropped, filtered_bgr, tesseract_cmd=options.tesseract_cmd)
        if target_rotate:
            cv_rot_map = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            cv_rot = cv_rot_map[target_rotate]
            filtered_bgr = cv2.rotate(filtered_bgr, cv_rot)
            steps["recortada"] = cv2.rotate(steps["recortada"], cv_rot)
            effective_rotation = (effective_rotation + target_rotate) % 360

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
