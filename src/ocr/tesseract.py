"""Tesseract OCR backend implementation.

Word-level Tesseract bounding boxes are aggregated into receipt lines with bounded
geometry and 1-based sequential line identifiers, satisfying the shared OcrEngine protocol.

Ref: Docs/Planning/implementation_plan.md, section 4.5.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytesseract

from src.core.types import BBox, OcrDocument, OcrLine

if TYPE_CHECKING:
    from src.core.settings import Settings

_TESSERACT_LOCK = threading.Lock()


class TesseractEngine:
    """Tesseract-based OCR engine aggregating word boxes into lines."""

    name = "tesseract"

    def __init__(
        self,
        lang: str = "por",
        psm: int = 6,
        tesseract_cmd: Path | str | None = None,
        image_to_data_fn: Any | None = None,
    ) -> None:
        """Configure Tesseract language, page segmentation mode, and executable path.

        Args:
            lang: Tesseract language code. Defaults to 'por'.
            psm: Tesseract page segmentation mode (0-13). Defaults to 6.
            tesseract_cmd: Optional explicit path to the tesseract executable.
            image_to_data_fn: Optional callable injection point for testing.

        Raises:
            ValueError: If language is empty or PSM mode is outside 0-13.
        """
        if not lang.strip():
            raise ValueError("Tesseract language code must not be empty.")
        if psm not in range(14):
            raise ValueError("Tesseract PSM mode must be between 0 and 13.")

        if tesseract_cmd is None:
            from src.core.settings import get_settings

            tesseract_cmd = get_settings().tesseract_cmd

        self.lang = lang
        self.psm = psm
        self.tesseract_cmd = str(tesseract_cmd) if tesseract_cmd is not None else None
        self._image_to_data_fn = image_to_data_fn

    def read(self, image: np.ndarray) -> OcrDocument:
        """Recognise text and aggregate word boxes into reading lines.

        Args:
            image: BGR or grayscale image, typically produced by preprocessing.

        Returns:
            OcrDocument containing lines sorted in reading order.

        Raises:
            ValueError: If image is empty or has unsupported channels.
            pytesseract.TesseractNotFoundError: If Tesseract executable is unavailable.
            pytesseract.TesseractError: If Tesseract inference fails.
        """
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Input image must not be empty or zero-sized.")
        if image.ndim == 2:
            image_for_tesseract = image
        elif image.ndim == 3 and image.shape[2] == 3:
            image_for_tesseract = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input image must be grayscale or have 3 BGR channels.")

        height, width = image.shape[:2]
        start_time = time.perf_counter()
        data = self._image_to_data(image_for_tesseract)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        required_fields = (
            "text",
            "conf",
            "left",
            "top",
            "width",
            "height",
            "block_num",
            "par_num",
            "line_num",
        )
        if any(field not in data for field in required_fields):
            raise ValueError("Tesseract output is missing required word fields.")

        field_lengths = {len(data[field]) for field in required_fields}
        if len(field_lengths) != 1:
            raise ValueError("Tesseract output fields have inconsistent lengths.")

        grouped_words: dict[tuple[int, int, int], list[tuple[str, float, int, int, int, int]]] = {}
        for index, raw_text in enumerate(data["text"]):
            text = str(raw_text).strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Tesseract returned invalid confidence at word {index}.") from error
            if confidence < 0:
                continue

            try:
                left = int(data["left"][index])
                top = int(data["top"][index])
                word_width = int(data["width"][index])
                word_height = int(data["height"][index])
                block_num = int(data["block_num"][index])
                par_num = int(data["par_num"][index])
                line_num = int(data["line_num"][index])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Tesseract returned invalid geometry at word {index}.") from error

            x1 = max(0, min(width, left))
            y1 = max(0, min(height, top))
            x2 = max(x1, min(width, left + word_width))
            y2 = max(y1, min(height, top + word_height))
            if x2 <= x1 or y2 <= y1:
                continue

            key = (block_num, par_num, line_num)
            grouped_words.setdefault(key, []).append((text, confidence, x1, y1, x2, y2))

        grouped_lines: list[tuple[int, int, tuple[int, int, int], OcrLine]] = []
        for key, words in grouped_words.items():
            x1 = min(word[2] for word in words)
            y1 = min(word[3] for word in words)
            x2 = max(word[4] for word in words)
            y2 = max(word[5] for word in words)
            line_text = " ".join(word[0] for word in sorted(words, key=lambda word: (word[2], word[3])))
            confidence = sum(word[1] for word in words) / len(words) / 100.0
            confidence = max(0.0, min(1.0, confidence))
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            grouped_lines.append((y1, x1, key, OcrLine(id=0, text=line_text, confidence=confidence, bbox=bbox)))

        grouped_lines.sort(key=lambda line: (line[0], line[1], line[2]))
        lines = [
            OcrLine(id=index + 1, text=line.text, confidence=line.confidence, bbox=line.bbox)
            for index, (_, _, _, line) in enumerate(grouped_lines)
        ]
        return OcrDocument(
            lines=lines,
            engine="tesseract",
            elapsed_ms=elapsed_ms,
            image_size=(width, height),
        )

    def _image_to_data(self, image: np.ndarray) -> dict[str, list[Any]]:
        """Call pytesseract while isolating process-wide executable setting."""
        if self._image_to_data_fn is not None:
            return self._image_to_data_fn(image)

        with _TESSERACT_LOCK:
            tesseract_module = pytesseract.pytesseract
            original_cmd = tesseract_module.tesseract_cmd
            try:
                if self.tesseract_cmd is not None:
                    tesseract_module.tesseract_cmd = self.tesseract_cmd
                result = pytesseract.image_to_data(
                    image,
                    lang=self.lang,
                    config=f"--psm {self.psm}",
                    output_type=pytesseract.Output.DICT,
                )
            finally:
                tesseract_module.tesseract_cmd = original_cmd
        return result


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Run Tesseract OCR on a receipt and print recognized numbered lines.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="Tesseract document reading preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Specific receipt image path")
    parser.add_argument("--lang", type=str, default="por", help="Tesseract language code")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode (0-13)")
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

    print("Initializing Tesseract engine...")
    engine = TesseractEngine(lang=args.lang, psm=args.psm, tesseract_cmd=settings.tesseract_cmd)

    print("Running OCR recognition...")
    doc = engine.read(image_to_read)

    print(f"OCR finished in {doc.elapsed_ms:.1f} ms. Lines found: {len(doc.lines)}\n")
    print("--- Extracted Text with Line Citations ---")
    print(doc.as_prompt_text())


if __name__ == "__main__":
    main()
