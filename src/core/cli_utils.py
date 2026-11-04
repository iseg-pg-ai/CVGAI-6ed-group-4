"""Shared CLI utilities for receipt loading and result formatting.

Ref: Docs/implementation_plan.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core.types import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import FieldValue


def list_receipt_images(receipts_dir: Path) -> list[Path]:
    """Return sorted receipt image paths inside a directory.

    Args:
        receipts_dir: Directory containing receipt images.

    Returns:
        Sorted list of image file paths matching IMAGE_EXTENSIONS.
    """
    if not receipts_dir.is_dir():
        return []
    return sorted(p for p in receipts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_and_load_receipt(
    file_path: str | Path | None,
    settings: Settings,
) -> tuple[Path, np.ndarray] | None:
    """Resolve target receipt path and load it as a BGR OpenCV image.

    Args:
        file_path: Optional explicit file path string or Path from CLI arguments.
        settings: Application settings containing default receipts directory.

    Returns:
        Tuple of (target_path, bgr_image) or None if resolution or loading failed.
    """
    target_path: Path | None = None
    if file_path:
        candidate = Path(file_path)
        if not candidate.is_file():
            print(f"Error: Specified file not found: {candidate}")
            return None
        target_path = candidate
    else:
        if not settings.receipts_dir.is_dir():
            print(f"Receipts directory not found: {settings.receipts_dir}")
            return None
        receipt_files = list_receipt_images(settings.receipts_dir)
        if not receipt_files:
            print(f"No receipt images found in {settings.receipts_dir}")
            return None
        target_path = receipt_files[0]

    print(f"Loading receipt: {target_path.name}")
    raw_image = cv2.imread(str(target_path))
    if raw_image is None:
        print(f"Error: Unable to read image file: {target_path}")
        return None

    return target_path, raw_image


def format_extraction_field(name: str, field: FieldValue) -> str:
    """Format a single extracted field for CLI display.

    Args:
        name: Canonical field name.
        field: Extracted field value container.

    Returns:
        A single printable line.
    """
    value = field.value if field.value is not None else "-"
    line = f"line {field.line_id}" if field.line_id is not None else "no line"
    return f"  {name:<16} {field.status:<11} {line:<9} {value}"
