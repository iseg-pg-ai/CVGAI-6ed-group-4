"""NIF pattern matching, merchant/customer disambiguation, and header zoning.

Ref: Docs/implementation_plan.md, section 5 (WP-09).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.core.normalizers import normalizar_entidade, parse_nif, validar_nif
from src.core.types import CONSUMIDOR_FINAL_NIF, FieldValue, OcrLine

#: Nine digits with optional single spaces or tabs between them.
_NIF_SCAN = re.compile(r"(?<!\d)\d(?:[ \t]*\d){8}(?!\d)")
NIF_SCAN = _NIF_SCAN

#: Words that identify the customer side of a receipt.
_CLIENT_MARKER = re.compile(r"\b(?:cliente|adquirente|consumidor|consumidora|n\s+c|cont\s+n\w*|contr|apelido)\b")
CLIENT_MARKER = _CLIENT_MARKER

#: Words that identify the issuer side of a receipt.
_MERCHANT_MARKER = re.compile(r"\b(?:nif|nipc|contribuinte|emitente|comerciante|empresa|crc)\b")
MERCHANT_MARKER = _MERCHANT_MARKER

#: Vertical share of the image considered the header, where the issuer NIF block lives.
_HEADER_BAND_RATIO = 0.4
HEADER_BAND_RATIO = _HEADER_BAND_RATIO


def fold_text(text: str) -> str:
    """Return the keyword-comparable form of a line.

    Reuses :func:`src.core.normalizers.normalizar_entidade`, which the project defines as
    "lowercase, no accents, no punctuation, whitespace collapsed".
    """
    return normalizar_entidade(text)


_fold = fold_text


def nifs_in(text: str) -> list[str]:
    """Find every nine-digit NIF in a line, in reading order.

    Args:
        text: Raw line text.

    Returns:
        Canonical nine-digit strings, including the consumer-final sentinel.
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


_nifs_in = nifs_in


def nif_field(nif: str | None, line: OcrLine | None) -> FieldValue:
    """Build the FieldValue for one NIF field.

    A nine-digit candidate that fails the check digit is reported as INVALIDO.
    """
    if nif is None or line is None:
        return FieldValue(value=None, status="AUSENTE")
    status = "ENCONTRADO" if validar_nif(nif) else "INVALIDO"
    return FieldValue(value=nif, status=status, line_id=line.id)


_nif_field = nif_field


def header_band(lines: Sequence[OcrLine], image_size: tuple[int, int]) -> list[OcrLine]:
    """Return the lines that make up the top header band of the document."""
    height = image_size[1]
    if height > 0:
        cutoff = height * _HEADER_BAND_RATIO
        band = [line for line in lines if line.bbox.y1 < cutoff]
        if band:
            return band
    return list(lines[: max(1, len(lines) // 3)])


_header_band = header_band


def extract_nifs(lines: Sequence[OcrLine], image_size: tuple[int, int]) -> tuple[FieldValue, FieldValue]:
    """Extract the issuer and customer NIFs without assigning the same number to both.

    Args:
        lines: Recognised lines in reading order.
        image_size: Size of the processed image as (width, height).

    Returns:
        The merchant field and the customer field, in that order.
    """
    merchant_pairs: list[tuple[str, OcrLine]] = []
    client_pairs: list[tuple[str, OcrLine]] = []
    bare_pairs: list[tuple[str, OcrLine]] = []

    for idx, line in enumerate(lines):
        folded = fold_text(line.text)
        nifs = [nif for nif in nifs_in(line.text) if nif != CONSUMIDOR_FINAL_NIF]
        if not nifs:
            continue

        has_preceding_client_context = False
        for back in range(1, min(9, idx + 1)):
            back_folded = fold_text(lines[idx - back].text)
            if re.search(r"\b(?:nome|vome|apelido|cliente|morada)\b", back_folded):
                has_preceding_client_context = True
                break

        has_client = (_CLIENT_MARKER.search(folded) is not None) or has_preceding_client_context
        if re.search(r"\b(?:apoio|suporte|linha)\b", folded):
            has_client = False
        has_merchant = _MERCHANT_MARKER.search(folded) is not None

        if has_client:
            valid_c = [n for n in nifs if validar_nif(n)]
            client_pairs.append((valid_c[-1] if valid_c else nifs[-1], line))
            if has_merchant and len(nifs) > 1:
                valid_m = [n for n in nifs[:-1] if validar_nif(n)]
                merchant_pairs.append((valid_m[0] if valid_m else nifs[0], line))
        elif has_merchant:
            valid_m = [n for n in nifs if validar_nif(n)]
            merchant_pairs.append((valid_m[0] if valid_m else nifs[0], line))
        else:
            bare_pairs.extend((nif, line) for nif in nifs)

    client_nif, client_line = client_pairs[0] if client_pairs else (None, None)

    if merchant_pairs:
        coll = [p for p in merchant_pairs if p[0][0] in "56" and validar_nif(p[0])]
        indiv = [p for p in merchant_pairs if p[0][0] in "123"]
        if client_nif is None and coll and indiv:
            merchant_nif, merchant_line = coll[0]
            client_nif, client_line = indiv[-1]
        elif coll:
            merchant_nif, merchant_line = coll[0]
        else:
            merchant_nif, merchant_line = merchant_pairs[0]
    else:
        header_ids = {line.id for line in header_band(lines, image_size)}
        fallback = [
            (nif, line) for nif, line in bare_pairs if line.id in header_ids and nif != client_nif and validar_nif(nif)
        ]
        merchant_nif, merchant_line = fallback[0] if fallback else (None, None)

    if merchant_nif is not None and merchant_nif[0] in "123" and client_nif is None:
        header_ids = {line.id for line in header_band(lines, image_size)}
        header_coll = [p for p in bare_pairs if p[1].id in header_ids and p[0][0] in "56"]
        if header_coll:
            client_nif, client_line = merchant_nif, merchant_line
            merchant_nif, merchant_line = header_coll[0]

    if client_nif is not None and client_nif == merchant_nif:
        client_nif, client_line = None, None

    return nif_field(merchant_nif, merchant_line), nif_field(client_nif, client_line)


_extract_nifs = extract_nifs
