"""OCR engines (EasyOCR and Tesseract) and engine factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import OcrEngine, OcrEngineName


def build_ocr_engine(
    engine_name: OcrEngineName | str | None = None,
    settings: Settings | None = None,
    *,
    gpu: bool = False,
    warmup: bool = False,
) -> OcrEngine:
    """Instantiate and configure an OCR engine based on name or settings.

    Args:
        engine_name: Optional explicit engine name ('easyocr' or 'tesseract').
            When omitted, defaults to settings.ocr_engine.
        settings: Optional custom Settings. When omitted, get_settings() is used.
        gpu: Enable GPU acceleration for EasyOCR.
        warmup: Run a zero-array warmup pass for EasyOCR.

    Returns:
        Configured OcrEngine instance.
    """
    from src.core.settings import get_settings

    cfg = settings or get_settings()
    choice = (engine_name or cfg.ocr_engine).strip().lower()
    if choice == "tesseract":
        from src.ocr.tesseract import TesseractEngine

        return TesseractEngine(lang="por", psm=6, tesseract_cmd=cfg.tesseract_cmd)

    from src.ocr.easyocr import EasyOCREngine

    return EasyOCREngine(langs=cfg.easyocr_langs, gpu=gpu, warmup=warmup)


__all__ = [
    "build_ocr_engine",
]
