"""Gradio interactive application for receipt reading, step-by-step CV pipeline, and grounded Q&A.

Assembles the full pipeline into three interactive tabs:
1. Document Reader and Grounded Q&A Assistant (live demo)
2. Step-by-Step Computer Vision Pipeline (didactic inspection for reports and presentation)
3. Validation and Benchmark Dashboard (accuracy metrics and error analysis)

Ref: Docs/Planning/implementation_plan.md, section 7, WP-16.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cv2
import gradio as gr
import numpy as np

from src.evidence import crop, draw_all_ocr_boxes, highlight
from src.extractor_llm import extract as extract_llm
from src.extractor_regex import extract as extract_regex
from src.llm_client import build_client
from src.ocr_engine import EasyOCREngine
from src.ocr_tesseract import TesseractEngine
from src.preprocessing import PreprocessOptions, PreprocessResult, preprocess
from src.qa_assistant import ask as ask_qa
from src.qr_reader import read as read_qr
from src.settings import get_settings
from src.types import IMAGE_EXTENSIONS, Extraction, OcrDocument, OcrEngine

if TYPE_CHECKING:
    from src.settings import Settings

# In-memory caches
_OCR_CACHE: dict[str, tuple[PreprocessResult, OcrDocument, dict[str, str] | None]] = {}
_ENGINE_CACHE: dict[str, OcrEngine] = {}


def clear_caches() -> None:
    """Clear in-memory OCR document and engine caches."""
    _OCR_CACHE.clear()
    _ENGINE_CACHE.clear()


def _from_gradio_rgb(image: np.ndarray) -> np.ndarray:
    """Convert Gradio RGB/RGBA/Grayscale array to 3-channel BGR for the OpenCV pipeline.

    Args:
        image: Source array passed by Gradio.

    Returns:
        3-channel BGR uint8 array.

    Raises:
        ValueError: If the input array has an unsupported shape.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Formato de imagem nao suportado com shape: {image.shape}")


def _to_gradio_rgb(image: np.ndarray | None) -> np.ndarray | None:
    """Convert OpenCV BGR array back to RGB for Gradio browser display.

    Args:
        image: Source BGR array from OpenCV pipeline or None.

    Returns:
        RGB array ready for Gradio display, or None.
    """
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
        f"{image_bgr.shape}:{image_bgr.dtype}:{opts.rotation_deg}:{opts.auto_crop}:"
        f"{opts.remove_shadows}:{opts.clahe_clip_limit}:{opts.clahe_tile_grid_size}:"
        f"{opts.auto_orient_osd}:{engine_name.lower()}".encode()
    )
    return hasher.hexdigest()


def _get_ocr_engine(engine_name: str, settings: Settings | None = None) -> OcrEngine:
    """Retrieve or instantiate a cached OCR engine instance."""
    key = engine_name.strip().lower()
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    cfg = settings or get_settings()
    engine: OcrEngine
    if key == "tesseract":
        engine = TesseractEngine(lang="por", psm=6, tesseract_cmd=cfg.tesseract_cmd)
    else:
        engine = EasyOCREngine(langs=cfg.easyocr_langs, warmup=False)

    _ENGINE_CACHE[key] = engine
    return engine


def _get_ocr_document(
    image_bgr: np.ndarray,
    opts: PreprocessOptions,
    engine_name: str,
    settings: Settings | None = None,
) -> tuple[PreprocessResult, OcrDocument, dict[str, str] | None]:
    """Retrieve preprocessed result, OCR document, and QR data from cache or compute them."""
    cache_key = _compute_cache_key(image_bgr, opts, engine_name)
    if cache_key in _OCR_CACHE:
        return _OCR_CACHE[cache_key]

    # Step 1: Preprocess (BGR -> BGR)
    prep_result = preprocess(image_bgr, opts)

    # Step 2: OCR on aligned/filtered BGR image
    engine = _get_ocr_engine(engine_name, settings=settings)
    ocr_doc = engine.read(prep_result.image)

    # Step 3: QR Code
    qr_data = read_qr(prep_result.image, raw_image=image_bgr)

    # Store in cache (limit size to 20 entries)
    if len(_OCR_CACHE) >= 20:
        _OCR_CACHE.pop(next(iter(_OCR_CACHE)))
    _OCR_CACHE[cache_key] = (prep_result, ocr_doc, qr_data)

    return prep_result, ocr_doc, qr_data


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
    np.ndarray | None,  # annotated_image (RGB for Gradio)
    np.ndarray | None,  # evidence_crop (RGB for Gradio)
    str,  # metadata_info
    list[list[str]],  # structured_table
    np.ndarray | None,  # step1_raw (RGB)
    np.ndarray | None,  # step2_cropped (RGB)
    np.ndarray | None,  # step3_filtered (RGB)
    np.ndarray | None,  # step4_boxes (RGB)
    str,  # ocr_prompt_text
]:
    """Run full preprocessing, OCR, extraction, and visual evidence highlighting."""
    if image is None:
        return None, None, "Nenhuma imagem carregada.", [], None, None, None, None, ""

    try:
        image_bgr = _from_gradio_rgb(image)
    except Exception as exc:
        return None, None, f"Erro na conversao da imagem: {exc}", [], None, None, None, None, ""

    rot_deg = "auto" if rotation_str == "auto" else int(rotation_str)
    opts = PreprocessOptions(
        rotation_deg=rot_deg,  # type: ignore[arg-type]
        auto_crop=auto_crop,
        remove_shadows=remove_shadows,
        clahe_clip_limit=clahe_clip,
    )

    try:
        prep_result, ocr_doc, qr_data = _get_ocr_document(image_bgr, opts, ocr_engine_name, settings=settings)
    except Exception as exc:
        err_msg = f"Erro no processamento de imagem/OCR: {exc}"
        return None, None, err_msg, [], None, None, None, None, ""

    # Map extractor choice with transparent fallback logging
    extraction: Extraction
    fallback_note = ""
    try:
        if extractor_name == "Regex Baseline":
            extraction = extract_regex(ocr_doc, qr=qr_data)
        elif extractor_name == "LLM Cloud (Gemini)":
            client = build_client(provider="llm_gemini", settings=settings)
            extraction = extract_llm(ocr_doc, qr=qr_data, client=client)
        elif extractor_name == "LLM Cloud (OpenAI)":
            client = build_client(provider="llm_openai", settings=settings)
            extraction = extract_llm(ocr_doc, qr=qr_data, client=client)
        else:
            client = build_client(provider="llm_local", settings=settings)
            extraction = extract_llm(ocr_doc, qr=qr_data, client=client)
    except Exception as exc:
        fallback_note = f" | Aviso: {extractor_name} indisponivel ({exc}); utilizado regex como recurso alternativo"
        extraction = extract_regex(ocr_doc, qr=qr_data)

    # Build Structured Fields Dataframe rows
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

    # Evidence rendering in BGR
    annotated_bgr = highlight(prep_result.image, ocr_doc, highlight_line_ids)
    evidence_crop_bgr = crop(prep_result.image, ocr_doc, highlight_line_ids)

    # Step-by-step pipeline views in BGR (Tab 2)
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
    str,  # status_str
    str,  # answer_text
    np.ndarray | None,  # annotated_image (RGB)
    np.ndarray | None,  # evidence_crop (RGB)
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

    rot_deg = "auto" if rotation_str == "auto" else int(rotation_str)
    opts = PreprocessOptions(
        rotation_deg=rot_deg,  # type: ignore[arg-type]
        auto_crop=auto_crop,
        remove_shadows=remove_shadows,
        clahe_clip_limit=clahe_clip,
    )

    try:
        prep_result, ocr_doc, _ = _get_ocr_document(image_bgr, opts, ocr_engine_name, settings=settings)
    except Exception as exc:
        return "AUSENTE NO DOCUMENTO", f"Erro ao processar imagem para Q&A: {exc}", None, None

    provider_map = {
        "LLM Local (llama.cpp)": "llm_local",
        "LLM Cloud (Gemini)": "llm_gemini",
        "LLM Cloud (OpenAI)": "llm_openai",
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


def load_benchmark_report(settings: Settings | None = None) -> tuple[str, list[str], list[list[str]]]:
    """Load benchmark metrics markdown and CSV if generated, or provide placeholder.

    Returns:
        Tuple of (markdown_summary, csv_headers, csv_data_rows).
    """
    cfg = settings or get_settings()
    results_dir = cfg.results_dir

    md_path = results_dir / "metrics.md"
    csv_path = results_dir / "metrics.csv"

    if md_path.is_file():
        md_content = md_path.read_text(encoding="utf-8")
    else:
        md_content = (
            "### Relatorio de Avaliacao de Desempenho (Pendente de Execucao)\n\n"
            "O benchmark oficial sobre o golden set pode ser executado com o comando:\n\n"
            "```bash\n"
            "uv run python -m src.evaluate\n"
            "```\n\n"
            "Os resultados discriminados por metodo (Regex vs LLM) e por grupo (direito vs Amarrotado) "
            "serao gerados em `results/metrics.md` e `results/metrics.csv`."
        )

    csv_headers: list[str] = []
    csv_rows: list[list[str]] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as f:
            reader = list(csv.reader(f))
            if reader:
                csv_headers = reader[0]
                csv_rows = reader[1:]

    return md_content, csv_headers, csv_rows


def build_app(settings_override: Settings | None = None) -> gr.Blocks:
    """Build the multi-tab Gradio application interface.

    Args:
        settings_override: Optional custom Settings. When None, get_settings() is used.

    Returns:
        Configured Gradio Blocks application.
    """
    settings = settings_override or get_settings()
    example_files: list[list[Any]] = []
    if settings.receipts_dir.is_dir():
        found = sorted(
            [str(p) for p in settings.receipts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        example_files = [[p] for p in found[:6]]

    with gr.Blocks(title="Leitor Inteligente de Documentos Visuais") as demo:
        gr.Markdown(
            "# Leitor Inteligente de Faturas e Taloes de Compra\n"
            "**CVGAI 6ª Edicao — Grupo 4** | Pipeline hibrido de Visao Computacional, OCR e Modelos de Linguagem"
        )

        with gr.Tabs():
            # -------------------------------------------------------------------------
            # TAB 1: Demonstracao Principal (Leitor + Assistente)
            # -------------------------------------------------------------------------
            with gr.Tab("Leitor e Assistente de Documentos"), gr.Row():
                # Left Column: Inputs & Controls
                with gr.Column(scale=4):
                    input_image = gr.Image(label="Fotografia do Talao", type="numpy")

                    with gr.Accordion("Parametros de Pre-processamento (OpenCV)", open=False):
                        rot_choice = gr.Radio(
                            choices=["auto", "0", "90", "180", "270"],
                            value="auto",
                            label="Rotacao (Orientacao)",
                        )
                        chk_crop = gr.Checkbox(value=True, label="Recorte Automatico do Documento")
                        chk_shadows = gr.Checkbox(value=True, label="Atenuacao de Sombras de Vincos")
                        slider_clahe = gr.Slider(
                            minimum=0.0, maximum=5.0, value=2.0, step=0.5, label="Intensidade de Contraste CLAHE"
                        )

                    with gr.Row():
                        ocr_choice = gr.Radio(
                            choices=["EasyOCR", "Tesseract"],
                            value="EasyOCR",
                            label="Motor OCR",
                        )
                        extractor_choice = gr.Radio(
                            choices=[
                                "Regex Baseline",
                                "LLM Local (llama.cpp)",
                                "LLM Cloud (Gemini)",
                                "LLM Cloud (OpenAI)",
                            ],
                            value="Regex Baseline",
                            label="Metodo de Extracao",
                        )

                    btn_process = gr.Button("Processar Documento", variant="primary")

                    if example_files:
                        gr.Examples(
                            examples=example_files,
                            inputs=[input_image],
                            label="Exemplos do Golden Set",
                        )

                # Center Column: Visual Grounding Evidence
                with gr.Column(scale=4):
                    output_annotated = gr.Image(label="Documento com Destaque Visual", type="numpy")
                    output_crop = gr.Image(label="Recorte Comprovativo da Evidencia", type="numpy")
                    txt_metadata = gr.Textbox(label="Metadados da Execucao", interactive=False)

                # Right Column: Extracted Fields & Grounded Q&A
                with gr.Column(scale=4):
                    gr.Markdown("### Dados Estruturados Extraidos")
                    tbl_fields = gr.Dataframe(
                        headers=["Campo", "Valor Extraido", "Estado", "Linha OCR", "QR Code AT"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        label="Campos Obrigatorios do Recibo",
                    )

                    gr.Markdown("### Assistente Verificavel de Perguntas e Respostas")
                    txt_question = gr.Textbox(
                        label="Pergunta em Linguagem Natural",
                        placeholder="Ex: Qual o NIF do cliente? Qual o total a pagar?",
                    )

                    with gr.Row():
                        btn_q1 = gr.Button("NIF do Cliente", size="sm")
                        btn_q2 = gr.Button("Metodo de Pagamento", size="sm")
                        btn_q3 = gr.Button("Email da Loja", size="sm")
                        btn_q4 = gr.Button("Total Pago", size="sm")

                    btn_ask = gr.Button("Perguntar ao Documento", variant="secondary")

                    with gr.Row():
                        txt_qa_status = gr.Textbox(label="Estado da Resposta", interactive=False, scale=1)
                        txt_qa_answer = gr.Textbox(label="Resposta do Assistente", lines=2, interactive=False, scale=3)

            # -------------------------------------------------------------------------
            # TAB 2: Pipeline Passo a Passo (Didatico / Slides)
            # -------------------------------------------------------------------------
            with gr.Tab("Pipeline Passo a Passo (Visao Computacional)"):
                gr.Markdown(
                    "### Evolucao Visual das Etapas de Pre-processamento e OCR\n"
                    "Permite capturar ecrans diretos para demonstrar o impacto do alinhamento, "
                    "recorte por contornos, remocao de sombras e CLAHE."
                )
                with gr.Row():
                    step1_img = gr.Image(label="1. Imagem Bruta", type="numpy")
                    step2_img = gr.Image(label="2. Alinhada e Recortada", type="numpy")
                with gr.Row():
                    step3_img = gr.Image(label="3. Filtrada e Otimizada (CLAHE)", type="numpy")
                    step4_img = gr.Image(label="4. Mapa de Caixas Delimitadoras (OCR)", type="numpy")

                with gr.Accordion("Texto Indexado por Linha (Prompt LLM)", open=False):
                    txt_ocr_prompt = gr.Textbox(
                        label="Texto Reconhecido com Identificadores Estaveis [NN]",
                        lines=12,
                        interactive=False,
                    )

            # -------------------------------------------------------------------------
            # TAB 3: Validacao e Benchmark
            # -------------------------------------------------------------------------
            with gr.Tab("Validacao e Benchmark"):
                gr.Markdown("### Metricas Quantitativas sobre o Golden Set")
                md_benchmark = gr.Markdown()
                tbl_benchmark = gr.Dataframe(
                    interactive=False,
                    label="Tabela de Metricas Detalhada (results/metrics.csv)",
                )
                btn_refresh_bench = gr.Button("Atualizar Relatorio de Metricas", size="sm")

        # -----------------------------------------------------------------------------
        # Wire Events
        # -----------------------------------------------------------------------------
        btn_process.click(
            fn=lambda img, rot, crop_chk, sh_chk, clahe, ocr_eng, extr: process_document(
                img, rot, crop_chk, sh_chk, clahe, ocr_eng, extr, settings=settings
            ),
            inputs=[
                input_image,
                rot_choice,
                chk_crop,
                chk_shadows,
                slider_clahe,
                ocr_choice,
                extractor_choice,
            ],
            outputs=[
                output_annotated,
                output_crop,
                txt_metadata,
                tbl_fields,
                step1_img,
                step2_img,
                step3_img,
                step4_img,
                txt_ocr_prompt,
            ],
        )

        btn_ask.click(
            fn=lambda q, img, rot, crop_chk, sh_chk, clahe, ocr_eng, prov: answer_question(
                q, img, rot, crop_chk, sh_chk, clahe, ocr_eng, prov, settings=settings
            ),
            inputs=[
                txt_question,
                input_image,
                rot_choice,
                chk_crop,
                chk_shadows,
                slider_clahe,
                ocr_choice,
                extractor_choice,
            ],
            outputs=[
                txt_qa_status,
                txt_qa_answer,
                output_annotated,
                output_crop,
            ],
        )

        # Quick question shortcut buttons
        btn_q1.click(lambda: "Qual o NIF do cliente?", outputs=txt_question)
        btn_q2.click(lambda: "Qual o metodo de pagamento?", outputs=txt_question)
        btn_q3.click(lambda: "Qual o email da loja?", outputs=txt_question)
        btn_q4.click(lambda: "Qual o valor total pago?", outputs=txt_question)

        # Benchmark tab load & refresh handlers
        def _refresh_benchmark() -> tuple[str, list[list[str]]]:
            md, _, rows = load_benchmark_report(settings=settings)
            return md, rows

        demo.load(fn=_refresh_benchmark, outputs=[md_benchmark, tbl_benchmark])
        btn_refresh_bench.click(fn=_refresh_benchmark, outputs=[md_benchmark, tbl_benchmark])

    return demo


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
