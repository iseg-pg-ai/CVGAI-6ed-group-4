"""Gradio interactive application for receipt reading, step-by-step CV pipeline, and grounded Q&A.

Assembles the full pipeline into three interactive tabs:
1. Document Reader and Grounded Q&A Assistant (live demo)
2. Step-by-Step Computer Vision Pipeline (didactic inspection for reports and presentation)
3. Validation and Benchmark Dashboard (accuracy metrics and error analysis)

Ref: Docs/implementation_plan.md, section 7, WP-16.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import gradio as gr
import numpy as np

from src.app_layout import build_blocks, load_benchmark_report
from src.core.settings import get_settings
from src.core.types import Extraction, OcrDocument, OcrEngine, RotationOption
from src.extraction.llm import extract as extract_llm
from src.extraction.llm_client import build_client
from src.extraction.qa_assistant import ask as ask_qa
from src.extraction.regex import extract as extract_regex
from src.ocr.easyocr import EasyOCREngine
from src.ocr.tesseract import TesseractEngine
from src.vision.evidence import crop, draw_all_ocr_boxes, highlight
from src.vision.preprocessing import PreprocessOptions, PreprocessResult, preprocess
from src.vision.qr_reader import read as read_qr

if TYPE_CHECKING:
    from src.core.settings import Settings

__all__ = [
    "EasyOCREngine",
    "PreprocessOptions",
    "TesseractEngine",
    "_CACHE_LOCK",
    "_ENGINE_CACHE",
    "_OCR_CACHE",
    "_build_preprocess_options",
    "_compute_cache_key",
    "_from_gradio_rgb",
    "_get_ocr_document",
    "_get_ocr_engine",
    "_to_gradio_rgb",
    "answer_question",
    "ask_qa",
    "build_app",
    "clear_caches",
    "crop",
    "draw_all_ocr_boxes",
    "extract_llm",
    "extract_regex",
    "highlight",
    "load_benchmark_report",
    "main",
    "preprocess",
    "process_document",
    "read_qr",
]

# In-memory thread-safe caches
_CACHE_LOCK = threading.Lock()
_OCR_CACHE: dict[str, tuple[PreprocessResult, OcrDocument, dict[str, str] | None]] = {}
_ENGINE_CACHE: dict[str, OcrEngine] = {}


def clear_caches() -> None:
    """Clear in-memory OCR document and engine caches."""
    with _CACHE_LOCK:
        _OCR_CACHE.clear()
        _ENGINE_CACHE.clear()


def _from_gradio_rgb(image: np.ndarray) -> np.ndarray:
    """Convert Gradio RGB/RGBA/Grayscale array to 3-channel BGR for the OpenCV pipeline."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Formato de imagem nao suportado com shape: {image.shape}")


def _to_gradio_rgb(image: np.ndarray | None) -> np.ndarray | None:
    """Convert OpenCV BGR array back to RGB for Gradio browser display."""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _compute_cache_key(image_bgr: np.ndarray, opts: PreprocessOptions, engine_name: str) -> str:
    """Compute deterministic SHA256 cache key for an image and processing configuration."""
    hasher = hashlib.sha256()
    hasher.update(image_bgr.tobytes())
    hasher.update(
        f"{image_bgr.shape}:{image_bgr.dtype}:{opts.max_side}:{opts.rotation_deg}:{opts.auto_crop}:"
        f"{opts.remove_shadows}:{opts.clahe_clip_limit}:{opts.clahe_tile_grid_size}:"
        f"{opts.auto_orient_osd}:{opts.auto_deskew}:{engine_name.lower()}".encode()
    )
    return hasher.hexdigest()


def _get_ocr_engine(engine_name: str, settings: Settings | None = None) -> OcrEngine:
    """Retrieve or instantiate a cached OCR engine instance."""
    key = engine_name.strip().lower()
    with _CACHE_LOCK:
        if key in _ENGINE_CACHE:
            return _ENGINE_CACHE[key]

    cfg = settings or get_settings()
    engine: OcrEngine
    if key == "tesseract":
        engine = TesseractEngine(lang="por", psm=6, tesseract_cmd=cfg.tesseract_cmd)
    else:
        engine = EasyOCREngine(langs=cfg.easyocr_langs, warmup=False)

    with _CACHE_LOCK:
        _ENGINE_CACHE[key] = engine
    return engine


def _get_ocr_document(
    image_bgr: np.ndarray,
    opts: PreprocessOptions,
    engine_name: str,
    settings: Settings | None = None,
) -> tuple[PreprocessResult, OcrDocument, dict[str, str] | None]:
    """Retrieve or compute the OCR document and AT QR payload with thread-safe caching."""
    cache_key = _compute_cache_key(image_bgr, opts, engine_name)
    with _CACHE_LOCK:
        if cache_key in _OCR_CACHE:
            return _OCR_CACHE[cache_key]

    prep_result = preprocess(image_bgr, opts=opts)
    engine = _get_ocr_engine(engine_name, settings=settings)
    ocr_doc = engine.read(prep_result.image)
    qr_data = read_qr(prep_result.image, raw_image=image_bgr)

    with _CACHE_LOCK:
        if len(_OCR_CACHE) >= 20:
            _OCR_CACHE.pop(next(iter(_OCR_CACHE)))
        _OCR_CACHE[cache_key] = (prep_result, ocr_doc, qr_data)

    return prep_result, ocr_doc, qr_data


def _build_preprocess_options(
    rotation_str: str,
    auto_crop: bool,
    remove_shadows: bool,
    clahe_clip: float,
    cfg: Settings,
) -> PreprocessOptions:
    """Build PreprocessOptions from Gradio UI controls and application settings."""
    rot_deg: RotationOption = "auto" if rotation_str == "auto" else int(rotation_str)  # type: ignore[assignment]
    return PreprocessOptions(
        max_side=cfg.max_side,
        rotation_deg=rot_deg,
        auto_crop=auto_crop,
        remove_shadows=remove_shadows,
        clahe_clip_limit=clahe_clip,
        tesseract_cmd=cfg.tesseract_cmd,
    )


def process_document(
    image: np.ndarray | None,
    rotation_str: str,
    auto_crop: bool,
    remove_shadows: bool,
    clahe_clip: float,
    ocr_engine_name: str,
    extractor_name: str,
    settings: Settings | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    str,
    list[list[str]],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    """Run full preprocessing, OCR, extraction, and visual evidence highlighting."""
    if image is None:
        return None, None, "Nenhuma imagem carregada.", [], None, None, None, None, ""

    try:
        image_bgr = _from_gradio_rgb(image)
    except Exception as exc:
        return None, None, f"Erro na conversao da imagem: {exc}", [], None, None, None, None, ""

    cfg = settings or get_settings()
    opts = _build_preprocess_options(rotation_str, auto_crop, remove_shadows, clahe_clip, cfg)

    try:
        prep_result, ocr_doc, qr_data = _get_ocr_document(image_bgr, opts, ocr_engine_name, settings=settings)
    except Exception as exc:
        err_msg = f"Erro no processamento de imagem/OCR: {exc}"
        return None, None, err_msg, [], None, None, None, None, ""

    extraction: Extraction
    fallback_note = ""
    try:
        if extractor_name == "Regex Baseline":
            extraction = extract_regex(ocr_doc, qr=qr_data)
        elif extractor_name == "LLM Cloud (Gemini)":
            client = build_client(provider="llm_gemini", settings=settings)
            extraction = extract_llm(ocr_doc, qr=qr_data, client=client)
        else:
            client = build_client(provider="llm_local", settings=settings)
            extraction = extract_llm(ocr_doc, qr=qr_data, client=client)
    except Exception as exc:
        fallback_note = f" | Aviso: {extractor_name} indisponivel ({exc}); utilizado regex como recurso alternativo"
        extraction = extract_regex(ocr_doc, qr=qr_data)

    table_rows: list[list[str]] = []
    qr_mapping = {
        "entidade": "-",
        "nif_comerciante": (qr_data.get("nif_comerciante") or "-") if qr_data else "-",
        "nif_cliente": (qr_data.get("nif_cliente") or "-") if qr_data else "-",
        "data": (qr_data.get("data") or "-") if qr_data else "-",
        "total": (qr_data.get("total") or "-") if qr_data else "-",
    }

    highlight_line_ids: list[int] = []
    for field_name, fv in extraction.fields.items():
        val_str = str(fv.value) if fv.value is not None else "-"
        line_ref = f"Linha {fv.line_id}" if fv.line_id is not None else "-"
        qr_val = qr_mapping.get(field_name, "-")
        table_rows.append([field_name, val_str, fv.status, line_ref, qr_val])
        if fv.line_id is not None:
            highlight_line_ids.append(fv.line_id)

    annotated_bgr = highlight(prep_result.image, ocr_doc, highlight_line_ids)
    evidence_crop_bgr = crop(prep_result.image, ocr_doc, highlight_line_ids)

    step1_raw_bgr = prep_result.steps.get("bruta", prep_result.image)
    step2_cropped_bgr = prep_result.steps.get("recortada", prep_result.image)
    step3_filtered_bgr = prep_result.steps.get("filtrada", prep_result.image)
    step4_boxes_bgr = draw_all_ocr_boxes(prep_result.image, ocr_doc)
    ocr_prompt_text = ocr_doc.as_prompt_text()

    qr_summary = "Detetado" if qr_data else "Nao detetado (recibo sem QR ou danificado)"
    metadata_info = (
        f"Motor OCR: {ocr_doc.engine} ({ocr_doc.elapsed_ms:.1f} ms, {len(ocr_doc.lines)} linhas) | "
        f"Extracao: {extraction.method} ({extraction.elapsed_ms:.1f} ms) | QR: {qr_summary}"
        f"{fallback_note}"
    )

    return (
        _to_gradio_rgb(annotated_bgr),
        _to_gradio_rgb(evidence_crop_bgr),
        metadata_info,
        table_rows,
        _to_gradio_rgb(step1_raw_bgr),
        _to_gradio_rgb(step2_cropped_bgr),
        _to_gradio_rgb(step3_filtered_bgr),
        _to_gradio_rgb(step4_boxes_bgr),
        ocr_prompt_text,
    )


def answer_question(
    question: str,
    image: np.ndarray | None,
    rotation_str: str,
    auto_crop: bool,
    remove_shadows: bool,
    clahe_clip: float,
    ocr_engine_name: str,
    provider_choice: str,
    settings: Settings | None = None,
) -> tuple[
    str,
    str,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Execute grounded question answering over the cached OCR document."""
    if image is None:
        return "AUSENTE NO DOCUMENTO", "Nenhum documento carregado para consulta.", None, None

    if not question.strip():
        return "AUSENTE NO DOCUMENTO", "Por favor introduza uma pergunta sobre o documento.", None, None

    try:
        image_bgr = _from_gradio_rgb(image)
    except Exception as exc:
        return "AUSENTE NO DOCUMENTO", f"Erro na conversao da imagem: {exc}", None, None

    cfg = settings or get_settings()
    opts = _build_preprocess_options(rotation_str, auto_crop, remove_shadows, clahe_clip, cfg)

    try:
        prep_result, ocr_doc, _ = _get_ocr_document(image_bgr, opts, ocr_engine_name, settings=settings)
    except Exception as exc:
        return "AUSENTE NO DOCUMENTO", f"Erro ao processar imagem para Q&A: {exc}", None, None

    provider_map = {
        "LLM Local (llama.cpp)": "llm_local",
        "LLM Cloud (Gemini)": "llm_gemini",
    }
    provider = provider_map.get(provider_choice, "llm_local")

    try:
        client = build_client(provider=provider, settings=settings)
        ans = ask_qa(question, ocr_doc, client=client)
    except Exception as exc:
        return "AUSENTE NO DOCUMENTO", f"Erro no assistente de linguagem: {exc}", None, None

    annotated_bgr = highlight(prep_result.image, ocr_doc, ans.line_ids)
    evidence_crop_bgr = crop(prep_result.image, ocr_doc, ans.line_ids)

    return (
        ans.status,
        ans.answer,
        _to_gradio_rgb(annotated_bgr),
        _to_gradio_rgb(evidence_crop_bgr),
    )


def build_app(settings_override: Settings | None = None) -> gr.Blocks:
    """Build the multi-tab Gradio application interface.

    Args:
        settings_override: Optional custom Settings. When None, get_settings() is used.

    Returns:
        Configured Gradio Blocks application.
    """
    settings = settings_override or get_settings()
    return build_blocks(
        process_document_fn=process_document,
        answer_question_fn=answer_question,
        report_fn=load_benchmark_report,
        settings=settings,
    )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Launch the Gradio server.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    parser = argparse.ArgumentParser(description="Launch Gradio receipt assistant app")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public gradio link")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    app = build_app(settings_override=settings_override)
    app.queue().launch(server_port=args.port, share=args.share, theme="soft")


if __name__ == "__main__":
    main()
