"""Gradio UI layout blocks, tab construction, and event wiring.

Ref: Docs/implementation_plan.md, section 7, WP-16.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gradio as gr

from src.core.cli_utils import list_receipt_images
from src.core.settings import get_settings
from src.core.types import ROTATION_CHOICES

if TYPE_CHECKING:
    from src.core.settings import Settings


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
            "uv run python -m src.evaluation.evaluate\n"
            "```\n\n"
            "Os resultados discriminados por metodo (Regex vs LLM) e por grupo (direito vs amarrotado) "
            "serao gerados em `results/metrics.md` e `results/metrics.csv`."
        )

    csv_headers: list[str] = []
    csv_rows: list[list[str]] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))
            if reader:
                csv_headers = reader[0]
                csv_rows = reader[1:]

    return md_content, csv_headers, csv_rows


def build_blocks(
    process_document_fn: Callable[..., Any],
    answer_question_fn: Callable[..., Any],
    report_fn: Callable[..., Any],
    settings: Settings,
) -> gr.Blocks:
    """Build the multi-tab Gradio interface with injected callbacks.

    Args:
        process_document_fn: Callback executing full OCR and extraction pipeline.
        answer_question_fn: Callback executing natural language Q&A.
        report_fn: Callback loading evaluation benchmark reports.
        settings: Active application configuration.

    Returns:
        Configured Gradio Blocks instance.
    """
    example_files: list[list[Any]] = [[str(p)] for p in list_receipt_images(settings.receipts_dir)]

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
                with gr.Column(scale=4):
                    input_image = gr.Image(label="Fotografia do Talao", type="numpy")

                    with gr.Accordion("Parametros de Pre-processamento (OpenCV)", open=False):
                        rot_choice = gr.Radio(
                            choices=list(ROTATION_CHOICES),
                            value="auto",
                            label="Rotacao (Orientacao)",
                        )
                        chk_crop = gr.Checkbox(value=True, label="Recorte Automatico do Documento")
                        chk_shadows = gr.Checkbox(value=True, label="Atenuacao de Sombras de Vincos")
                        slider_clahe = gr.Slider(
                            minimum=0.0, maximum=5.0, value=2.0, step=0.5, label="Intensidade de Contraste CLAHE"
                        )
                        slider_gamma = gr.Slider(
                            minimum=0.5, maximum=2.5, value=1.0, step=0.1, label="Densidade da Tinta (Gamma)"
                        )
                        slider_sharpen = gr.Slider(
                            minimum=0.0, maximum=3.0, value=0.35, step=0.05, label="Nitidez (Unsharp Mask)"
                        )
                        slider_shadow_k = gr.Slider(
                            minimum=11, maximum=51, value=25, step=2, label="Janela de Sombra (Shadow Kernel)"
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

                with gr.Column(scale=4):
                    output_annotated = gr.Image(label="Documento com Destaque Visual", type="numpy")
                    output_crop = gr.Image(label="Recorte Comprovativo da Evidencia", type="numpy")
                    txt_metadata = gr.Textbox(label="Metadados da Execucao", lines=2, interactive=False)

                with gr.Column(scale=4):
                    gr.Markdown("### Dados Estruturados Extraidos")
                    tbl_fields = gr.Dataframe(
                        headers=["Campo", "Valor Extraido", "Estado", "Linha OCR", "QR Code AT"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        label="Campos Obrigatorios do Recibo",
                    )

                    gr.Markdown("### Assistente Verificavel de Perguntas e Respostas")
                    qa_provider_choice = gr.Radio(
                        choices=[
                            "LLM Local (llama.cpp)",
                            "LLM Cloud (Gemini)",
                        ],
                        value="LLM Local (llama.cpp)",
                        label="Modelo LLM para Q&A",
                    )
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
                    headers=[
                        "filename",
                        "grupo",
                        "campo",
                        "esperado",
                        "obtido",
                        "acertou",
                        "falso_positivo",
                        "omissao",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Tabela de Metricas Detalhada (results/metrics.csv)",
                )
                btn_refresh_bench = gr.Button("Atualizar Relatorio de Metricas", size="sm")

        # -----------------------------------------------------------------------------
        # Wire Events
        # -----------------------------------------------------------------------------
        btn_process.click(
            fn=lambda img, rot, crop_chk, sh_chk, clahe, gamma, sharpen, shadow_k, ocr_eng, extr: process_document_fn(
                img, rot, crop_chk, sh_chk, clahe, gamma, sharpen, shadow_k, ocr_eng, extr, settings=settings
            ),
            inputs=[
                input_image,
                rot_choice,
                chk_crop,
                chk_shadows,
                slider_clahe,
                slider_gamma,
                slider_sharpen,
                slider_shadow_k,
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
            fn=lambda q, img, rot, crop_chk, sh_chk, clahe, gamma, sharpen, shadow_k, ocr_eng, prov: answer_question_fn(
                q, img, rot, crop_chk, sh_chk, clahe, gamma, sharpen, shadow_k, ocr_eng, prov, settings=settings
            ),
            inputs=[
                txt_question,
                input_image,
                rot_choice,
                chk_crop,
                chk_shadows,
                slider_clahe,
                slider_gamma,
                slider_sharpen,
                slider_shadow_k,
                ocr_choice,
                qa_provider_choice,
            ],
            outputs=[
                txt_qa_status,
                txt_qa_answer,
                output_annotated,
                output_crop,
            ],
        )

        btn_q1.click(lambda: "Qual o NIF do cliente?", outputs=txt_question)
        btn_q2.click(lambda: "Qual o metodo de pagamento?", outputs=txt_question)
        btn_q3.click(lambda: "Qual o email da loja?", outputs=txt_question)
        btn_q4.click(lambda: "Qual o valor total pago?", outputs=txt_question)

        def _refresh_benchmark() -> tuple[str, Any]:
            md, headers, rows = report_fn(settings=settings)
            if headers:
                return md, gr.update(value=rows, headers=headers)
            return md, gr.update(value=[])

        demo.load(fn=_refresh_benchmark, outputs=[md_benchmark, tbl_benchmark])
        btn_refresh_bench.click(fn=_refresh_benchmark, outputs=[md_benchmark, tbl_benchmark])

    return demo
