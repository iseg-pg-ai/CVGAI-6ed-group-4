"""Visual evidence rendering for receipt text grounding.

Draws OCR spatial bounding boxes, highlights active answer evidence lines with semi-transparent
colored overlays, and extracts padded zoomed image snippets supporting extracted fields or Q&A.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-11.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.types import IMAGE_EXTENSIONS, OcrDocument, OcrLine

if TYPE_CHECKING:
    from src.settings import Settings


def _draw_label_badge(
    image: np.ndarray,
    text: str,
    x: int,
    y1: int,
    bg_color: tuple[int, int, int] = (40, 40, 40),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw a compact, high-contrast label tag above or inside a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position label above box if space permits; otherwise draw inside the top of the box
    label_y = y1 - 3 if y1 - 3 > th + 3 else y1 + th + 3
    x_clamped = max(0, min(image.shape[1] - tw - 4, x))

    cv2.rectangle(
        image,
        (x_clamped, label_y - th - 2),
        (x_clamped + tw + 4, label_y + baseline + 1),
        bg_color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x_clamped + 2, label_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_all_ocr_boxes(
    image: np.ndarray,
    doc: OcrDocument,
    box_color: tuple[int, int, int] = (255, 120, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Render all recognized OCR lines with bounding boxes and [id] labels.

    Used by Tab 2 Step 4 of the Gradio interface to inspect full spatial OCR detections.

    Args:
        image: Source BGR or grayscale image.
        doc: OcrDocument produced by an OCR engine.
        box_color: BGR color for bounding rectangle borders.
        text_color: BGR color for label text.
        thickness: Line width of bounding boxes.

    Returns:
        New BGR image copy with all bounding boxes and [id] tags drawn.

    Raises:
        ValueError: If the input image is empty or has zero area.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must not be empty or zero-sized.")

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    for line in doc.lines:
        b = line.bbox
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), box_color, thickness)
        _draw_label_badge(canvas, f"[{line.id}]", b.x1, b.y1, bg_color=(40, 40, 40), text_color=text_color)

    return canvas


def highlight(
    image: np.ndarray,
    doc: OcrDocument,
    line_ids: Sequence[int] | set[int],
    highlight_color: tuple[int, int, int] = (0, 200, 255),
    muted_color: tuple[int, int, int] = (190, 190, 190),
    alpha: float = 0.25,
) -> np.ndarray:
    """Highlight cited evidence lines with semi-transparent overlays while muting unselected lines.

    Draws all unselected OCR lines in muted_color and cited line_ids in highlight_color
    with a semi-transparent fill overlay and a crisp border.

    Args:
        image: Source BGR or grayscale image.
        doc: OcrDocument containing all detected lines.
        line_ids: Identifiers of the lines supporting the active answer or field.
        highlight_color: BGR color for cited evidence lines.
        muted_color: BGR color for inactive lines.
        alpha: Opacity of the semi-transparent highlight fill (0.0 to 1.0).

    Returns:
        New BGR image copy with visual highlights applied.

    Raises:
        ValueError: If the input image is empty or has zero area.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must not be empty or zero-sized.")

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    target_set = set(line_ids)

    # 1. Draw inactive lines with subtle, thin border
    for line in doc.lines:
        if line.id not in target_set:
            b = line.bbox
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), muted_color, 1)

    matching_lines = [line for line in doc.lines if line.id in target_set]
    if not matching_lines:
        return canvas

    # 2. Semi-transparent fill overlay on cited lines
    overlay = canvas.copy()
    for line in matching_lines:
        b = line.bbox
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), highlight_color, -1)

    alpha_clamped = max(0.0, min(1.0, alpha))
    cv2.addWeighted(overlay, alpha_clamped, canvas, 1.0 - alpha_clamped, 0, canvas)

    # 3. Solid borders and distinct label badges for cited lines
    for line in matching_lines:
        b = line.bbox
        cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), highlight_color, 2)
        _draw_label_badge(canvas, f"[{line.id}]", b.x1, b.y1, bg_color=highlight_color, text_color=(0, 0, 0))

    return canvas


def crop(
    image: np.ndarray,
    doc: OcrDocument,
    line_ids: Sequence[int] | set[int],
    pad_px: int = 16,
) -> np.ndarray | None:
    """Extract a cropped image slice enclosing the union bounding box of cited lines.

    Used by Tab 1 of the Gradio interface to display the zoomed evidence snippet.

    Args:
        image: Source BGR or grayscale image.
        doc: OcrDocument containing detected lines.
        line_ids: Identifiers of the lines to crop.
        pad_px: Pixel margin added around the bounding box.

    Returns:
        Cropped BGR image slice or None if line_ids contains no matching lines.

    Raises:
        ValueError: If the input image is empty or has zero area.
    """
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image must not be empty or zero-sized.")

    target_set = set(line_ids)
    matching_lines: list[OcrLine] = [line for line in doc.lines if line.id in target_set]
    if not matching_lines:
        return None

    img_h, img_w = image.shape[:2]

    min_x = min(line.bbox.x1 for line in matching_lines)
    min_y = min(line.bbox.y1 for line in matching_lines)
    max_x = max(line.bbox.x2 for line in matching_lines)
    max_y = max(line.bbox.y2 for line in matching_lines)

    x1 = max(0, min_x - pad_px)
    y1 = max(0, min_y - pad_px)
    x2 = min(img_w, max_x + pad_px)
    y2 = min(img_h, max_y + pad_px)

    if x2 <= x1 or y2 <= y1:
        return None

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    return bgr[y1:y2, x1:x2].copy()


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to preview highlight and crop rendering on receipts.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from PIL import Image

    from src.ocr_engine import EasyOCREngine
    from src.preprocessing import PreprocessOptions, preprocess
    from src.settings import get_settings

    parser = argparse.ArgumentParser(description="Receipt visual evidence preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument(
        "--lines",
        "-l",
        type=str,
        default=None,
        help="Comma-separated line IDs to highlight and crop, e.g. '1,2,3'",
    )
    parser.add_argument("--save", "-s", type=str, default=None, help="Save preview image to disk")

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

    print("Preprocessing receipt...")
    prep_result = preprocess(raw_image, PreprocessOptions())

    print("Running OCR...")
    engine = EasyOCREngine(langs=settings.easyocr_langs, warmup=True)
    doc = engine.read(prep_result.image)

    # Determine lines to highlight
    if args.lines:
        line_ids = [int(x.strip()) for x in args.lines.split(",") if x.strip().isdigit()]
    else:
        # Default: pick lines matching common keywords or top 2 lines
        picked = [
            line_item.id
            for line_item in doc.lines
            if any(k in line_item.text.upper() for k in ("TOTAL", "NIF", "NIPC", "CONTRIB", "CONTRIBUINTE"))
        ]
        line_ids = picked[:3] or [1, 2] if doc.lines else []

    print(f"Highlighting line IDs: {line_ids}")
    annotated = highlight(prep_result.image, doc, line_ids)
    evidence_crop = crop(prep_result.image, doc, line_ids)

    if evidence_crop is not None:
        print(f"Extracted evidence crop size: {evidence_crop.shape[1]}x{evidence_crop.shape[0]} px")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"Saved highlighted image to: {save_path}")
    else:
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).show(title=f"Evidencia ({target_path.name})")


if __name__ == "__main__":
    main()
