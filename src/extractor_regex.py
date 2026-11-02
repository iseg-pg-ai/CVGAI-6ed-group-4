"""Rule-based baseline extractor for the five required receipt fields.

This module implements approach A of ``PROJECT_PLAN.md`` (section 4, level 2): manual regular
expressions over the OCR text, plus algorithmic NIF validation. It is deliberately naive and
is meant to be beaten by the LLM extractors. Every heuristic is named and documented so the
report can point at the exact rule that breaks on a given document instead of hand-waving.

Three behaviours are worth stating up front, because they are easy to get wrong:

- The AT payload is accepted by :func:`extract` for interface parity with the LLM extractor
  and the application, but it is never used to populate a field. The baseline has to be
  measured on the OCR text alone, and payment-terminal slips carry no QR at all.
- The consumer-final sentinel (``999999990``) is a structurally valid NIF: it passes the
  modulo-11 check digit. It is therefore rejected by an explicit constant and not by
  :func:`src.normalizers.validar_nif`.
- The total is read with this module's own money pattern rather than
  :func:`src.normalizers.parse_moeda`, because the latter returns the *first* number in a
  string and would read ``"TOTAL 2 UN 16,10"`` as ``2``.

Ref: Docs/Planning/implementation_plan.md, section 5 (WP-09).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from src.normalizers import limpar_texto, normalizar_entidade, parse_data, parse_moeda, parse_nif, validar_nif
from src.types import (
    CONSUMIDOR_FINAL_NIF,
    EARLIEST_PLAUSIBLE_DATE,
    IMAGE_EXTENSIONS,
    Extraction,
    FieldValue,
    OcrDocument,
    OcrLine,
)

if TYPE_CHECKING:
    from src.settings import Settings

#: Nine digits with optional single spaces or tabs between them.
#:
#: Same shape as the pattern inside :func:`src.normalizers.parse_nif`, repeated here because
#: that helper only ever returns the first match and one receipt line can carry two NIFs (the
#: merchant's and the customer's). Canonicalisation still goes through ``parse_nif``.
_NIF_SCAN = re.compile(r"(?<!\d)\d(?:[ \t]*\d){8}(?!\d)")

#: Words that identify the customer side of a receipt.
_CLIENT_MARKER = re.compile(r"\b(?:cliente|adquirente|consumidor|consumidora)\b")

#: Words that identify the issuer side of a receipt. Bare ``contribuinte`` belongs here: on
#: Portuguese receipts it labels the merchant's own NIF far more often than the customer's.
_MERCHANT_MARKER = re.compile(r"\b(?:nif|nipc|contribuinte|emitente|comerciante|empresa)\b")

#: Words that disqualify a line from being the merchant trading name.
_ENTITY_STOPWORDS = re.compile(
    r"\b(?:fatura|factura|recibo|talao|contribuinte|nif|nipc|atcud|cliente|total|iva|"
    r"troco|documento|original|duplicado|processado|certificado|software|operador|"
    r"data|hora|telefone|tel|www|http|email|desconto)\b"
)

#: A total line: an explicit total label followed closely by the amount, which must close the
#: line (optionally followed by a currency marker).
#:
#: The anchor on the end of the line is what keeps product lines such as ``TOTAL CARE 4,50``
#: and breakdown lines such as ``TOTAL IVA 3,02`` out of the result. ``SUBTOTAL`` never
#: matches because there is no word boundary between ``SUB`` and ``TOTAL``.
_TOTAL_LINE = re.compile(
    r"\b(?:VALOR\s+)?TOTAL(?:\s+A\s+PAGAR)?\s*[\s:.]*"
    r"(?P<amount>\d{1,3}(?:[ \u00a0\u202f.]\d{3})*[.,]\d{2}|\d+[.,]\d{2})"
    r"\s*(?:EUR|EUROS|€)?\s*$",
    re.IGNORECASE,
)

#: Minimum number of alphabetic characters for a candidate trading name.
_MIN_ENTITY_LETTERS = 3

#: Minimum share of alphabetic characters for a candidate trading name. Address and code
#: lines fall below this ratio.
_MIN_ENTITY_ALPHA_RATIO = 0.5

#: Vertical share of the image considered the header, where the issuer NIF block lives.
_HEADER_BAND_RATIO = 0.4


def _fold(text: str) -> str:
    """Return the keyword-comparable form of a line.

    Reuses :func:`src.normalizers.normalizar_entidade`, which the project defines as
    "lowercase, no accents, no punctuation, whitespace collapsed". That is exactly what word
    matching needs, and it avoids a second canonicalisation routine that could drift from it.

    Args:
        text: Raw line text.

    Returns:
        The folded line, or an empty string when there is nothing left after folding.
    """
    return normalizar_entidade(text)


def _nifs_in(text: str) -> list[str]:
    """Find every nine-digit NIF in a line, in reading order.

    Args:
        text: Raw line text.

    Returns:
        Canonical nine-digit strings. The consumer-final sentinel is returned too; rejecting
        it is the caller's decision, because the two fields treat it differently.
    """
    found: list[str] = []
    for match in _NIF_SCAN.finditer(text):
        candidate = match.group()
        if candidate.strip() == CONSUMIDOR_FINAL_NIF:
            found.append(CONSUMIDOR_FINAL_NIF)
            continue
        nif = parse_nif(candidate)
        if nif is not None:
            found.append(nif)
    return found


def _nif_field(nif: str | None, line: OcrLine | None) -> FieldValue:
    """Build the :class:`FieldValue` for one NIF field.

    A nine-digit candidate that fails the check digit is reported as ``INVALIDO`` and keeps
    its digits, so that scoring can count it as a wrong value rather than as an omission.

    Args:
        nif: Canonical nine-digit candidate, or ``None`` when nothing was found.
        line: Line that supports the candidate, or ``None``.

    Returns:
        The field value with its status and evidence.
    """
    if nif is None or line is None:
        return FieldValue(value=None, status="AUSENTE")
    status = "ENCONTRADO" if validar_nif(nif) else "INVALIDO"
    return FieldValue(value=nif, status=status, line_id=line.id)


def _extract_entity(lines: Sequence[OcrLine]) -> FieldValue:
    """Pick the merchant trading name.

    Heuristic: the first line that looks like a name, i.e. it has at least three letters, is
    more than half alphabetic, does not contain a document or field keyword, and either has
    two words or one word of at least five letters. On Portuguese receipts the issuer name is
    printed before the address, so "first plausible line" is the rule, not "most plausible".

    Args:
        lines: Recognised lines in reading order.

    Returns:
        The trading name as printed, or ``AUSENTE``.
    """
    for line in lines:
        text = limpar_texto(line.text)
        folded = _fold(text)
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


def _extract_nifs(lines: Sequence[OcrLine], image_size: tuple[int, int]) -> tuple[FieldValue, FieldValue]:
    """Extract the issuer and customer NIFs without assigning the same number to both.

    Resolution order, documented because it is a judgement call:

    1. The merchant NIF is the first one on a line carrying an issuer label (``NIF``,
       ``NIPC``, ``CONTRIBUINTE``, ...) and no customer label.
    2. Failing that, it is the first unlabelled NIF in the header band, which is where the
       issuer block lives on every layout in the corpus.
    3. The customer NIF is the last one on a line carrying a customer label. Taking the last
       is what makes a single line such as ``NIF: 500 100 144 NIF CLIENTE: 123 456 789``
       work: the merchant number comes before the label and the customer number after it.
    4. A customer candidate identical to the merchant NIF is discarded: a receipt identifies
       its issuer, so when only one number exists it belongs to the merchant.

    The consumer-final sentinel is skipped in both fields, so a receipt issued to a final
    consumer reports ``AUSENTE`` rather than ``ENCONTRADO 999999990`` or ``INVALIDO``.

    Args:
        lines: Recognised lines in reading order.
        image_size: Size of the processed image, used to locate the header band.

    Returns:
        The merchant field and the customer field, in that order.
    """
    merchant_pairs: list[tuple[str, OcrLine]] = []
    client_pairs: list[tuple[str, OcrLine]] = []
    bare_pairs: list[tuple[str, OcrLine]] = []

    for line in lines:
        folded = _fold(line.text)
        nifs = [nif for nif in _nifs_in(line.text) if nif != CONSUMIDOR_FINAL_NIF]
        if not nifs:
            continue

        has_client = _CLIENT_MARKER.search(folded) is not None
        has_merchant = _MERCHANT_MARKER.search(folded) is not None

        if has_client:
            client_pairs.append((nifs[-1], line))
            if has_merchant and len(nifs) > 1:
                merchant_pairs.append((nifs[0], line))
        elif has_merchant:
            merchant_pairs.append((nifs[0], line))
        else:
            bare_pairs.extend((nif, line) for nif in nifs)

    client_nif, client_line = client_pairs[0] if client_pairs else (None, None)

    if merchant_pairs:
        merchant_nif, merchant_line = merchant_pairs[0]
    else:
        header_ids = {line.id for line in _header_band(lines, image_size)}
        fallback = [(nif, line) for nif, line in bare_pairs if line.id in header_ids and nif != client_nif]
        merchant_nif, merchant_line = fallback[0] if fallback else (None, None)

    if client_nif is not None and client_nif == merchant_nif:
        client_nif, client_line = None, None

    return _nif_field(merchant_nif, merchant_line), _nif_field(client_nif, client_line)


def _header_band(lines: Sequence[OcrLine], image_size: tuple[int, int]) -> list[OcrLine]:
    """Return the lines that make up the top of the document.

    Geometry is preferred over line count because the number of lines varies wildly between a
    restaurant slip and a supermarket bill. The line-count fallback covers documents built
    without a meaningful image size, which happens in tests.

    Args:
        lines: Recognised lines in reading order.
        image_size: Size of the processed image as ``(width, height)``.

    Returns:
        The lines whose top edge falls inside the header band, or the first third of the
        document when geometry is unavailable.
    """
    height = image_size[1]
    if height > 0:
        cutoff = height * _HEADER_BAND_RATIO
        band = [line for line in lines if line.bbox.y1 < cutoff]
        if band:
            return band
    return list(lines[: max(1, len(lines) // 3)])


def _is_plausible_date(parsed: date) -> bool:
    """Reject dates a receipt cannot carry.

    Hyphenated dates are read as ``YY-MM-DD`` by :func:`src.normalizers.parse_data`, so a
    receipt printed as ``DD-MM-YY`` can be misread into the next decade. An issue date in the
    future is impossible, which makes the upper bound a cheap way to drop those misreads and
    fall through to the next candidate line.

    Args:
        parsed: Date returned by the normaliser.

    Returns:
        Whether the date is inside the plausible window.
    """
    return EARLIEST_PLAUSIBLE_DATE <= parsed <= date.today()


def _extract_date(lines: Sequence[OcrLine]) -> FieldValue:
    """Pick the first plausible issue date, in ISO form.

    Args:
        lines: Recognised lines in reading order.

    Returns:
        The date as ``YYYY-MM-DD``, or ``AUSENTE``.
    """
    for line in lines:
        parsed = parse_data(line.text)
        if parsed is None or not _is_plausible_date(parsed):
            continue
        return FieldValue(value=parsed.isoformat(), status="ENCONTRADO", line_id=line.id)

    return FieldValue(value=None, status="AUSENTE")


def _extract_total(lines: Sequence[OcrLine]) -> FieldValue:
    """Pick the amount due.

    The last line matching :data:`_TOTAL_LINE` wins, because the amount due is printed after
    its own breakdown. Zero and negative totals are reported as ``INVALIDO``: the ground-truth
    contract requires a strictly positive total, so a zero is a misread and not an answer.

    Args:
        lines: Recognised lines in reading order.

    Returns:
        The total as a two-decimal string, or ``AUSENTE``.
    """
    chosen: tuple[OcrLine, str] | None = None
    for line in lines:
        match = _TOTAL_LINE.search(limpar_texto(line.text))
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


def extract(doc: OcrDocument, qr: dict[str, str] | None = None) -> Extraction:
    """Extract the five required fields from an OCR document with regular expressions.

    Args:
        doc: Recognised document, produced by either OCR engine.
        qr: Optional AT payload. Accepted for interface parity and intentionally not read;
            see the module docstring.

    Returns:
        The extraction, tagged with ``method="regex"`` and the wall-clock duration.
    """
    started = time.perf_counter()

    merchant_nif, client_nif = _extract_nifs(doc.lines, doc.image_size)
    return Extraction(
        entidade=_extract_entity(doc.lines),
        nif_comerciante=merchant_nif,
        nif_cliente=client_nif,
        data=_extract_date(doc.lines),
        total=_extract_total(doc.lines),
        method="regex",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _format_field(name: str, field: FieldValue) -> str:
    """Render one field as an aligned row for the command-line preview.

    Args:
        name: Column name of the field.
        field: Field value to render.

    Returns:
        A single printable line.
    """
    value = field.value if field.value is not None else "-"
    line = f"line {field.line_id}" if field.line_id is not None else "no line"
    return f"  {name:<16} {field.status:<11} {line:<9} {value}"


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to inspect the regex baseline on one receipt.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.
        settings_override: Optional custom Settings. When None, ``get_settings()`` is used.
    """
    import cv2

    from src.ocr_engine import EasyOCREngine
    from src.ocr_tesseract import TesseractEngine
    from src.preprocessing import PreprocessOptions, preprocess
    from src.settings import get_settings

    parser = argparse.ArgumentParser(description="Regex baseline extractor preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument(
        "--engine",
        type=str,
        choices=("easyocr", "tesseract"),
        default=None,
        help="OCR engine to use (defaults to the configured one)",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    target_path: Path | None = None
    if args.file:
        candidate = Path(args.file)
        if not candidate.is_file():
            print(f"Error: Specified file not found: {candidate}")
            return
        target_path = candidate
    else:
        if not settings.receipts_dir.is_dir():
            print(f"Receipts directory not found: {settings.receipts_dir}")
            return
        receipt_files = sorted(
            [p for p in settings.receipts_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not receipt_files:
            print(f"No receipt images found in {settings.receipts_dir}")
            return
        target_path = receipt_files[0]

    print(f"Loading receipt: {target_path.name}")
    raw_image = cv2.imread(str(target_path))
    if raw_image is None:
        print(f"Error: Unable to read image file: {target_path}")
        return

    if args.no_preprocess:
        image_to_read = raw_image
    else:
        print("Preprocessing receipt...")
        image_to_read = preprocess(raw_image, PreprocessOptions()).image

    engine_name = args.engine or settings.ocr_engine
    engine = TesseractEngine() if engine_name == "tesseract" else EasyOCREngine(langs=settings.easyocr_langs)

    print(f"Running OCR with {engine.name}...")
    doc = engine.read(image_to_read)

    extraction = extract(doc)

    print(f"\n--- Regex baseline ({len(doc.lines)} lines, {extraction.elapsed_ms:.2f} ms) ---")
    for name, field in extraction.fields.items():
        print(_format_field(name, field))


if __name__ == "__main__":
    main()
