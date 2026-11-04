"""Rule-based baseline extractor for the five required receipt fields.

This module implements approach A of ``PROJECT_PLAN.md`` (section 4, level 2): manual regular
expressions over the OCR text, plus algorithmic NIF validation. It is deliberately naive and
is meant to be beaten by the LLM extractors. Every heuristic is named and documented so the
report can point at the exact rule that breaks on a given document instead of hand-waving.

Ref: Docs/implementation_plan.md, section 5 (WP-09).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

from src.core.cli_utils import format_extraction_field
from src.core.normalizers import (
    limpar_texto,
    parse_data,
    parse_moeda,
    validar_data,
)
from src.core.types import (
    CONSUMIDOR_FINAL_NIF,
    OCR_ENGINES,
    Extraction,
    FieldValue,
    OcrDocument,
    OcrLine,
)
from src.extraction.regex_nifs import (
    _CLIENT_MARKER,
    _HEADER_BAND_RATIO,
    _MERCHANT_MARKER,
    _NIF_SCAN,
    CLIENT_MARKER,
    HEADER_BAND_RATIO,
    MERCHANT_MARKER,
    NIF_SCAN,
    _extract_nifs,
    _fold,
    _header_band,
    _nif_field,
    _nifs_in,
    extract_nifs,
    fold_text,
    header_band,
    nif_field,
    nifs_in,
)

if TYPE_CHECKING:
    from src.core.settings import Settings

__all__ = [
    "CLIENT_MARKER",
    "CONSUMIDOR_FINAL_NIF",
    "HEADER_BAND_RATIO",
    "MERCHANT_MARKER",
    "NIF_SCAN",
    "_CLIENT_MARKER",
    "_ENTITY_STOPWORDS",
    "_HEADER_BAND_RATIO",
    "_MERCHANT_MARKER",
    "_MIN_ENTITY_ALPHA_RATIO",
    "_MIN_ENTITY_LETTERS",
    "_NIF_SCAN",
    "_TOTAL_LINE",
    "_extract_date",
    "_extract_entity",
    "_extract_nifs",
    "_extract_total",
    "_fold",
    "_header_band",
    "_nif_field",
    "_nifs_in",
    "extract",
    "extract_date",
    "extract_entity",
    "extract_nifs",
    "extract_total",
    "fold_text",
    "header_band",
    "main",
    "nif_field",
    "nifs_in",
]

#: Words that disqualify a line from being the merchant trading name.
_ENTITY_STOPWORDS = re.compile(
    r"\b(?:fatura|factura|recibo|talao|contribuinte|nif|nipc|atcud|cliente|total|iva|"
    r"troco|documento|original|duplicado|processado|certificado|software|operador|"
    r"data|hora|telefone|tel|www|http|email|desconto|bem|vindo|copia|segunda|via)\b"
)

#: A total line: an explicit total label followed closely by the amount.
_TOTAL_LINE = re.compile(
    r"\b(?:(?:VALOR\s+)?(?:TOTAL|[TI17][O0]T[A4]L)(?:\s+(?:(?:A\s+)?PAGAR|PAGO|LIQUIDO|GERAL|FINAL))?|(?:VALOR\s+)?A\s+PAGAR)\s*[\s:.]*"
    r"(?:EUR|EUROS|€)?\s*"
    r"(?P<amount>\d{1,3}(?:[ \u00a0\u202f.]\d{3})*[.,]\d{2}|\d+[.,]\d{2})"
    r"\s*(?:EUR|EUROS|€)?\s*$",
    re.IGNORECASE,
)

_MIN_ENTITY_LETTERS = 3
_MIN_ENTITY_ALPHA_RATIO = 0.5


def extract_entity(lines: Sequence[OcrLine], image_size: tuple[int, int] | None = None) -> FieldValue:
    """Pick the merchant trading name from the top header band of the document."""
    if not lines:
        return FieldValue(value=None, status="AUSENTE")

    candidate_lines = lines
    if image_size is not None and image_size[1] > 0:
        cutoff = image_size[1] * _HEADER_BAND_RATIO
        header_lines = [line for line in lines if line.bbox.y1 < cutoff]
        if header_lines:
            candidate_lines = header_lines

    for line in candidate_lines:
        text = limpar_texto(line.text)
        folded = fold_text(text)
        if not folded or _ENTITY_STOPWORDS.search(folded):
            continue

        letters = sum(1 for character in text if character.isalpha())
        visible = sum(1 for character in text if not character.isspace())
        if letters < _MIN_ENTITY_LETTERS or visible == 0:
            continue
        if letters / visible < _MIN_ENTITY_ALPHA_RATIO:
            continue

        words = folded.split()
        if len(words) >= 2 or len(words[0]) >= 5:
            return FieldValue(value=text, status="ENCONTRADO", line_id=line.id)

    return FieldValue(value=None, status="AUSENTE")


_extract_entity = extract_entity


def extract_date(lines: Sequence[OcrLine]) -> FieldValue:
    """Pick the first plausible issue date, in ISO form."""
    for line in lines:
        parsed = parse_data(line.text)
        if parsed is None or not validar_data(parsed):
            continue
        return FieldValue(value=parsed.isoformat(), status="ENCONTRADO", line_id=line.id)

    return FieldValue(value=None, status="AUSENTE")


_extract_date = extract_date


def extract_total(lines: Sequence[OcrLine]) -> FieldValue:
    """Pick the amount due from the last matching total line."""
    chosen: tuple[OcrLine, str] | None = None
    for line in lines:
        cleaned = re.sub(r"(\d)\s*([.,])\s*(\d)", r"\1\2\3", limpar_texto(line.text))
        match = _TOTAL_LINE.search(cleaned)
        if match is not None:
            chosen = (line, match.group("amount"))

    if chosen is None:
        return FieldValue(value=None, status="AUSENTE")

    line, token = chosen
    amount = parse_moeda(token)
    if amount is None:
        return FieldValue(value=None, status="AUSENTE")

    value = f"{amount:.2f}"
    if amount <= Decimal("0"):
        return FieldValue(value=value, status="INVALIDO", line_id=line.id)

    return FieldValue(value=value, status="ENCONTRADO", line_id=line.id)


_extract_total = extract_total


def extract(doc: OcrDocument, qr: dict[str, str] | None = None) -> Extraction:
    """Extract the five required fields from an OCR document with regular expressions."""
    started = time.perf_counter()

    merchant_nif, client_nif = extract_nifs(doc.lines, doc.image_size)
    return Extraction(
        entidade=extract_entity(doc.lines, doc.image_size),
        nif_comerciante=merchant_nif,
        nif_cliente=client_nif,
        data=extract_date(doc.lines),
        total=extract_total(doc.lines),
        method="regex",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to inspect the regex baseline on one receipt."""
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.ocr import build_ocr_engine
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="Regex baseline extractor preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument(
        "--engine",
        type=str,
        choices=OCR_ENGINES,
        default=None,
        help="OCR engine to use (defaults to the configured one)",
    )
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
        image_to_read = preprocess(raw_image, PreprocessOptions()).image

    engine = build_ocr_engine(args.engine, settings)

    print(f"Running OCR with {engine.name}...")
    doc = engine.read(image_to_read)

    extraction = extract(doc)

    print(f"\n--- Regex baseline ({len(doc.lines)} lines, {extraction.elapsed_ms:.2f} ms) ---")
    for name, field in extraction.fields.items():
        print(format_extraction_field(name, field))


if __name__ == "__main__":
    main()
