"""Run the receipt benchmark from image preprocessing through metric reporting.

Ref: Docs/Planning/implementation_plan.md, section 6, WP-15.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from src.types import EXTRACTION_METHODS, OCR_ENGINES, Extraction, GroundTruthRecord, MatchResult, MethodName, OcrEngine

if TYPE_CHECKING:
    from src.llm_client import LLMClient
    from src.preprocessing import PreprocessOptions
    from src.settings import Settings


def avaliar_registo(
    registo: GroundTruthRecord,
    receipts_dir: Path,
    ocr_engine: OcrEngine,
    *,
    method: MethodName = "regex",
    llm_client: LLMClient | None = None,
    preprocess_opts: PreprocessOptions | None = None,
) -> tuple[Extraction, list[MatchResult]]:
    """Run preprocessing, OCR, QR reading, extraction, and scoring for a single receipt.

    Args:
        registo: Ground-truth reference record for one receipt.
        receipts_dir: Directory containing the receipt image files.
        ocr_engine: Initialised OCR engine instance (EasyOCR or Tesseract).
        method: Extraction method ('regex', 'llm_local', 'llm_gemini', or 'llm_openai').
        llm_client: Optional injected LLM client (used when method != 'regex').
        preprocess_opts: Optional OpenCV preprocessing options. When None, default options apply.

    Returns:
        Tuple of (Extraction, list of 5 MatchResult objects).

    Raises:
        FileNotFoundError: If the receipt image does not exist.
        ValueError: If OpenCV cannot decode the receipt image.
    """
    import cv2

    from src.extractor_regex import extract as extract_regex
    from src.preprocessing import PreprocessOptions, preprocess
    from src.qr_reader import read as read_qr
    from src.scoring import pontuar_registo

    image_path = receipts_dir / registo.filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Receipt image not found: {image_path}")
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise ValueError(f"Unable to decode receipt image: {image_path}")

    prep = preprocess(raw_image, preprocess_opts or PreprocessOptions())
    doc = ocr_engine.read(prep.image)
    qr_data = read_qr(prep.image, raw_image=raw_image)

    if method == "regex":
        extraction = extract_regex(doc, qr=qr_data)
    else:
        from src.extractor_llm import extract as extract_llm
        from src.llm_client import build_client

        client = llm_client or build_client(method)
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

    Returns:
        Combined list of ``MatchResult`` objects across all evaluated receipts (5 per receipt).
    """
    if method != "regex" and llm_client is None:
        from src.llm_client import build_client

        llm_client = build_client(method)

    all_results: list[MatchResult] = []
    for record in registos:
        _, matches = avaliar_registo(
            record,
            receipts_dir,
            ocr_engine,
            method=method,
            llm_client=llm_client,
            preprocess_opts=preprocess_opts,
        )
        all_results.extend(matches)

    if destino_resultados is not None:
        from src.scoring import escrever_tabelas

        escrever_tabelas(all_results, destino_resultados)
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
    from src.golden_set import GoldenSetError, carregar_ground_truth
    from src.ocr_engine import EasyOCREngine
    from src.ocr_tesseract import TesseractEngine
    from src.preprocessing import PreprocessOptions
    from src.scoring import agregar
    from src.settings import get_settings

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
    parser.add_argument("--no-preprocess", action="store_true", help="Disable all preprocessing stages")
    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    try:
        records = carregar_ground_truth(args.csv, receipts_dir=args.receipts_dir)
    except GoldenSetError as error:
        for problem in error.errors:
            print(problem)
        return

    if args.engine == "tesseract":
        ocr_engine: OcrEngine = TesseractEngine(tesseract_cmd=settings.tesseract_cmd)
    else:
        ocr_engine = EasyOCREngine(langs=settings.easyocr_langs)

    preprocess_opts = (
        PreprocessOptions(
            rotation_deg=0,
            auto_crop=False,
            remove_shadows=False,
            clahe_clip_limit=0.0,
            auto_orient_osd=False,
        )
        if args.no_preprocess
        else None
    )
    results = avaliar_corpus(
        records,
        args.receipts_dir,
        ocr_engine,
        method=args.method,
        preprocess_opts=preprocess_opts,
        destino_resultados=args.output_dir,
    )
    metrics = agregar(results)["global"]
    print(f"Taxa de acerto global: {metrics['taxa_acerto']:.4f}")
    print(f"Taxa de falsos positivos: {metrics['taxa_falso_positivo']:.4f}")
    print(f"Taxa de omissão: {metrics['taxa_omissao']:.4f}")
    print(f"Relatórios: {args.output_dir / 'metrics.csv'} e {args.output_dir / 'metrics.md'}")


if __name__ == "__main__":
    main()
