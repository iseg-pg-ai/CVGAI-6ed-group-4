"""Run the receipt benchmark from image preprocessing through metric reporting.

Ref: Docs/implementation_plan.md, section 6, WP-15.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import (
    EXTRACTION_METHODS,
    OCR_ENGINES,
    Extraction,
    GroundTruthRecord,
    MatchResult,
    MethodName,
    OcrEngine,
)

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.extraction.llm_client import LLMClient
    from src.vision.preprocessing import PreprocessOptions


def avaliar_registo(
    registo: GroundTruthRecord,
    receipts_dir: Path,
    ocr_engine: OcrEngine,
    *,
    method: MethodName = "regex",
    llm_client: LLMClient | None = None,
    preprocess_opts: PreprocessOptions | None = None,
    settings: Settings | None = None,
) -> tuple[Extraction, list[MatchResult]]:
    """Run preprocessing, OCR, QR reading, extraction, and scoring for a single receipt.

    Args:
        registo: Ground-truth reference record for one receipt.
        receipts_dir: Directory containing the receipt image files.
        ocr_engine: Initialised OCR engine instance (EasyOCR or Tesseract).
        method: Extraction method ('regex', 'llm_local', or 'llm_gemini').
        llm_client: Optional injected LLM client (used when method != 'regex').
        preprocess_opts: Optional OpenCV preprocessing options. When None, default options apply.
        settings: Optional Settings instance for configuring preprocessing or client fallback.

    Returns:
        Tuple of (Extraction, list of 5 MatchResult objects).

    Raises:
        FileNotFoundError: If the receipt image does not exist.
        ValueError: If OpenCV cannot decode the receipt image.
    """
    import cv2

    from src.evaluation.scoring import pontuar_registo
    from src.extraction.regex import extract as extract_regex
    from src.vision.preprocessing import PreprocessOptions, preprocess
    from src.vision.qr_reader import read as read_qr

    image_path = receipts_dir / registo.filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Receipt image not found: {image_path}")
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise ValueError(f"Unable to decode receipt image: {image_path}")

    opts = preprocess_opts
    if opts is None:
        opts = (
            PreprocessOptions(
                max_side=settings.max_side,
                clahe_clip_limit=settings.clahe_clip_limit,
                tesseract_cmd=settings.tesseract_cmd,
            )
            if settings is not None
            else PreprocessOptions()
        )

    prep = preprocess(raw_image, opts)
    doc = ocr_engine.read(prep.image)

    # Fiscal keyword orientation verification: upside-down receipts produce unreadable OCR text
    # with almost zero receipt keywords. If flipping 180 produces substantially more keywords,
    # adopt the corrected upright orientation.
    fiscal_keywords = (
        "total",
        "fatura",
        "factura",
        "recibo",
        "nif",
        "nipc",
        "data",
        "iva",
        "cliente",
        "contribuinte",
        "atcud",
        "artigo",
        "subtotal",
    )
    current_text = " ".join(line.text.lower() for line in doc.lines)
    current_kw_count = sum(1 for kw in fiscal_keywords if kw in current_text)
    if current_kw_count <= 2:
        flipped_img = cv2.rotate(prep.image, cv2.ROTATE_180)
        flipped_doc = ocr_engine.read(flipped_img)
        flipped_text = " ".join(line.text.lower() for line in flipped_doc.lines)
        flipped_kw_count = sum(1 for kw in fiscal_keywords if kw in flipped_text)
        if flipped_kw_count >= 3 and flipped_kw_count > current_kw_count + 2:
            from src.vision.preprocessing import PreprocessResult

            prep = PreprocessResult(
                image=flipped_img,
                steps={**prep.steps, "recortada": flipped_img, "filtrada": flipped_img},
                rotation_deg=(prep.rotation_deg + 180) % 360,
            )
            doc = flipped_doc

    qr_data = read_qr(prep.image, raw_image=raw_image)

    if method == "regex":
        extraction = extract_regex(doc, qr=qr_data)
    else:
        from src.extraction.llm import extract as extract_llm
        from src.extraction.llm_client import build_client

        client = llm_client or build_client(method, settings=settings)
        extraction = extract_llm(doc, qr=qr_data, client=client)

    matches = pontuar_registo(registo, extraction)
    return extraction, matches


def avaliar_corpus(
    registos: Sequence[GroundTruthRecord],
    receipts_dir: Path,
    ocr_engine: OcrEngine,
    *,
    method: MethodName = "regex",
    llm_client: LLMClient | None = None,
    preprocess_opts: PreprocessOptions | None = None,
    destino_resultados: Path | None = None,
    settings: Settings | None = None,
    progress_callback: Callable[[int, int, GroundTruthRecord], None] | None = None,
) -> list[MatchResult]:
    """Evaluate validated records and optionally write detailed and aggregate reports.

    Args:
        registos: Validated ground-truth records to evaluate.
        receipts_dir: Directory containing the receipt images.
        ocr_engine: Initialised OCR engine instance, reused across all receipts.
        method: Extraction method to benchmark.
        llm_client: Optional injected LLM client, reused for all receipts.
        preprocess_opts: Optional OpenCV preprocessing options.
        destino_resultados: When provided, writes ``metrics.csv`` and ``metrics.md`` there.
        settings: Optional Settings instance forwarded to record evaluation.
        progress_callback: Optional callback invoked as ``callback(index, total, record)``
            before processing each receipt.

    Returns:
        Combined list of ``MatchResult`` objects across all evaluated receipts (5 per receipt).
    """
    if method != "regex" and llm_client is None:
        from src.extraction.llm_client import build_client

        llm_client = build_client(method, settings=settings)

    all_results: list[MatchResult] = []
    total_records = len(registos)
    for index, record in enumerate(registos, start=1):
        if progress_callback is not None:
            progress_callback(index, total_records, record)
        _, matches = avaliar_registo(
            record,
            receipts_dir,
            ocr_engine,
            method=method,
            llm_client=llm_client,
            preprocess_opts=preprocess_opts,
            settings=settings,
        )
        all_results.extend(matches)

    if destino_resultados is not None:
        from src.evaluation.scoring import escrever_tabelas

        preproc_active = (
            "desativado"
            if preprocess_opts is not None
            and not preprocess_opts.auto_crop
            and not preprocess_opts.remove_shadows
            and preprocess_opts.clahe_clip_limit <= 0.0
            else "ativo"
        )
        metadata = {
            "Motor OCR": getattr(ocr_engine, "name", str(ocr_engine)),
            "Método de extração": method,
            "Pré-processamento": preproc_active,
            "Documentos avaliados": str(len(registos)),
        }
        escrever_tabelas(all_results, destino_resultados, metadata=metadata)

    return all_results


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Load the golden set, run the configured benchmark, and print its aggregate rates.

    Args:
        argv: Optional CLI arguments, primarily for tests.
        settings_override: Optional custom Settings instance.
    """
    from src.core.settings import get_settings
    from src.evaluation.golden_set import GoldenSetError, carregar_ground_truth
    from src.evaluation.scoring import agregar
    from src.extraction.llm import LLMResponseError
    from src.extraction.llm_client import LLMError
    from src.ocr import build_ocr_engine
    from src.vision.preprocessing import PreprocessOptions

    settings = settings_override or get_settings()
    parser = argparse.ArgumentParser(description="Evaluate receipt extraction against the golden set")
    parser.add_argument("--csv", type=Path, default=settings.ground_truth_path, help="Ground-truth CSV path")
    parser.add_argument("--receipts-dir", type=Path, default=settings.receipts_dir, help="Receipt image directory")
    parser.add_argument("--output-dir", type=Path, default=settings.results_dir, help="Metrics output directory")
    parser.add_argument(
        "--engine",
        choices=OCR_ENGINES,
        default=settings.ocr_engine,
        help="OCR engine to use",
    )
    parser.add_argument(
        "--method",
        choices=EXTRACTION_METHODS,
        default="regex",
        help="Extraction method to benchmark",
    )
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of receipts to evaluate")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference for EasyOCR")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable all preprocessing stages")
    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    try:
        records = carregar_ground_truth(args.csv, receipts_dir=args.receipts_dir)
    except GoldenSetError as error:
        for problem in error.errors:
            print(problem)
        return

    print(f"Golden set carregado: {len(records)} registos.")
    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]
        print(f"Limite ativo: a avaliar apenas os primeiros {len(records)} registos.")

    print(f"A inicializar motor OCR ({args.engine})...")
    ocr_engine = build_ocr_engine(args.engine, settings, gpu=args.gpu)

    preprocess_opts = (
        PreprocessOptions(
            max_side=settings.max_side,
            rotation_deg=0,
            auto_crop=False,
            remove_shadows=False,
            clahe_clip_limit=0.0,
            auto_orient_osd=False,
            tesseract_cmd=settings.tesseract_cmd,
        )
        if args.no_preprocess
        else PreprocessOptions(
            max_side=settings.max_side,
            clahe_clip_limit=settings.clahe_clip_limit,
            tesseract_cmd=settings.tesseract_cmd,
        )
    )

    print(f"A iniciar benchmark ({len(records)} documentos, motor: {args.engine}, método: {args.method})...\n")

    def _show_progress(idx: int, total: int, record: GroundTruthRecord) -> None:
        print(f"[{idx:02d}/{total:02d}] A avaliar {record.filename} ({record.grupo})...", flush=True)

    try:
        results = avaliar_corpus(
            records,
            args.receipts_dir,
            ocr_engine,
            method=args.method,
            preprocess_opts=preprocess_opts,
            destino_resultados=args.output_dir,
            settings=settings,
            progress_callback=_show_progress,
        )
    except (LLMError, LLMResponseError) as err:
        print(f"Erro na extração com modelo de linguagem: {err}")
        return
    except Exception as err:
        print(f"Erro na execução do benchmark: {err}")
        return

    aggregated = agregar(results)
    metrics = aggregated["global"]
    print("\n--- Resultados do Benchmark ---")
    print(f"Taxa de acerto global: {metrics['taxa_acerto']:.4f}")
    print(f"Taxa de falsos positivos: {metrics['taxa_falso_positivo']:.4f}")
    print(f"Taxa de omissão: {metrics['taxa_omissao']:.4f}")
    if "por_grupo" in aggregated:
        for group, group_metrics in aggregated["por_grupo"].items():
            print(f"Taxa de acerto ({group}): {group_metrics['taxa_acerto']:.4f}")
    print(f"Relatórios: {args.output_dir / 'metrics.csv'} e {args.output_dir / 'metrics.md'}")


if __name__ == "__main__":
    main()
