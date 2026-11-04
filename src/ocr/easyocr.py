"""EasyOCR implementation of the OcrEngine protocol.

Extracts text lines and bounding boxes from preprocessed receipt images, clamping coordinates
to image bounds, filtering empty tokens, and assigning stable sequential 1-based identifiers.

Ref: Docs/implementation_plan.md, section 7, WP-05.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.core.types import BBox, OcrDocument, OcrLine, validate_image

# Suppress upstream EasyOCR/PyTorch internal deprecation and CPU-pin warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="easyocr.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.*")

if TYPE_CHECKING:
    from src.core.settings import Settings


class EasyOCREngine:
    """EasyOCR engine satisfying the OcrEngine protocol.

    Wraps easyocr.Reader behind a uniform interface and allows dependency injection of
    mock readers for deterministic, offline testing without downloading model weights.

    Attributes:
        name: Name of the engine ('easyocr').
        langs: Tuple of language codes loaded in the reader.
        gpu: Whether GPU acceleration is enabled.
    """

    name: str = "easyocr"

    def __init__(
        self,
        langs: tuple[str, ...] = ("pt", "en"),
        gpu: bool = False,
        warmup: bool = True,
        reader: Any | None = None,
    ) -> None:
        """Initialise the EasyOCR engine.

        Args:
            langs: Tuple of language identifiers, e.g. ('pt', 'en').
            gpu: Whether to run inference on GPU. Defaults to False for reproducibility.
            warmup: Whether to run a dummy inference at startup to pay weight-loading cost.
            reader: Optional pre-configured reader instance for testing or dependency injection.
        """
        self.langs = langs
        self.gpu = gpu

        if reader is not None:
            self._reader = reader
        else:
            import easyocr

            self._reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
            if warmup:
                dummy = np.full((64, 64, 3), 255, dtype=np.uint8)
                self._reader.readtext(dummy)

    def read(self, image: np.ndarray) -> OcrDocument:
        """Recognise text in an aligned, preprocessed receipt image.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            OcrDocument containing recognized lines with sequential 1-based IDs and bounding boxes.

        Raises:
            ValueError: If the input image is empty or has zero area.
        """
        validate_image(image)

        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        height, width = bgr.shape[:2]

        start_time = time.perf_counter()
        raw_results = self._reader.readtext(bgr, detail=1)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        candidates: list[tuple[int, int, str, float, BBox]] = []

        for item in raw_results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            poly, raw_text, conf = item[0], item[1], item[2]
            text = str(raw_text).strip()
            if not text:
                continue

            try:
                all_x = [pt[0] for pt in poly]
                all_y = [pt[1] for pt in poly]
            except (TypeError, IndexError):
                continue

            if not all_x or not all_y:
                continue

            x1 = max(0, min(width, int(round(min(all_x)))))
            y1 = max(0, min(height, int(round(min(all_y)))))
            x2 = max(x1, min(width, int(round(max(all_x)))))
            y2 = max(y1, min(height, int(round(max(all_y)))))

            # Discard zero-area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            confidence = max(0.0, min(1.0, float(conf)))
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            candidates.append((y1, x1, text, confidence, bbox))

        # Sort initial fragments top-to-bottom
        candidates.sort(key=lambda c: (c[4].y1, c[4].x1))

        # Group fragments that fall on the same horizontal line
        line_groups: list[list[tuple[int, int, str, float, BBox]]] = []
        for cand in candidates:
            cand_bbox = cand[4]
            cand_cy = (cand_bbox.y1 + cand_bbox.y2) / 2.0
            cand_h = max(1, cand_bbox.height)

            placed = False
            for group in line_groups:
                grp_y1 = min(item[4].y1 for item in group)
                grp_y2 = max(item[4].y2 for item in group)
                grp_cy = (grp_y1 + grp_y2) / 2.0
                grp_h = max(1, grp_y2 - grp_y1)

                overlap = min(cand_bbox.y2, grp_y2) - max(cand_bbox.y1, grp_y1)
                is_same_line = overlap > 0.45 * min(cand_h, grp_h) or abs(cand_cy - grp_cy) <= 0.45 * min(cand_h, grp_h)
                if is_same_line:
                    group.append(cand)
                    placed = True
                    break

            if not placed:
                line_groups.append([cand])

        # Sort lines top-to-bottom by top-most y coordinate
        line_groups.sort(key=lambda grp: min(item[4].y1 for item in grp))

        # Merge horizontal fragments left-to-right within each line
        lines: list[OcrLine] = []
        for idx, grp in enumerate(line_groups, start=1):
            grp.sort(key=lambda item: item[4].x1)
            line_text = " ".join(item[2] for item in grp)
            avg_conf = sum(item[3] for item in grp) / len(grp)
            line_bbox = BBox(
                x1=min(item[4].x1 for item in grp),
                y1=min(item[4].y1 for item in grp),
                x2=max(item[4].x2 for item in grp),
                y2=max(item[4].y2 for item in grp),
            )
            lines.append(OcrLine(id=idx, text=line_text, confidence=avg_conf, bbox=line_bbox))

        return OcrDocument(
            lines=lines,
            engine="easyocr",
            elapsed_ms=elapsed_ms,
            image_size=(width, height),
        )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Run OCR on a receipt and print recognized numbered lines.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="EasyOCR document reading preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Specific receipt image path")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference")
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

    print("Initializing EasyOCR engine...")
    engine = EasyOCREngine(langs=settings.easyocr_langs, gpu=args.gpu, warmup=True)

    print("Running OCR recognition...")
    doc = engine.read(image_to_read)

    print(f"OCR finished in {doc.elapsed_ms:.1f} ms. Lines found: {len(doc.lines)}\n")
    print("--- Extracted Text with Line Citations ---")
    print(doc.as_prompt_text())


if __name__ == "__main__":
    main()
