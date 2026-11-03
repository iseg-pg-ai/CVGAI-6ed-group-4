"""Functions to normalise and compare receipt field values.

Ref: Docs/Planning/implementation_plan.md, section 4.3.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

VALID_NIF_FIRST_DIGITS = frozenset("12356789")


def limpar_texto(raw: str) -> str:
    """Strip control characters, collapse whitespace, and trim extremities.

    Args:
        raw: Text to clean.

    Returns:
        Cleaned text with single spaces between words and no control characters.
    """
    characters: list[str] = []
    for character in raw:
        if character.isspace():
            characters.append(" ")
        elif unicodedata.category(character) != "Cc":
            characters.append(character)
    return " ".join("".join(characters).split())


def parse_moeda(raw: str) -> Decimal | None:
    """Parse the first interpretable monetary value in the text.

    Commas are treated as decimal separators. A dot followed by exactly three digits
    is treated as a thousands separator, as are grouped spaces.

    Args:
        raw: Text that may contain a monetary value and currency marker.

    Returns:
        Parsed Decimal amount or None if no valid amount is present.
    """
    match = re.search(r"(?<!\d)[+-]?\d+(?:[ \u00a0\u202f.,]\d+)*(?!\d)", limpar_texto(raw))
    if match is None:
        return None

    token = match.group().replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    comma_index = token.rfind(",")
    dot_index = token.rfind(".")
    decimal_separator: str | None = None

    if comma_index >= 0 and dot_index >= 0:
        decimal_separator = "," if comma_index > dot_index else "."
    elif comma_index >= 0:
        decimal_separator = ","
    elif dot_index >= 0:
        digits_after_dot = len(token) - dot_index - 1
        if digits_after_dot != 3:
            decimal_separator = "."

    if decimal_separator is None:
        normalized = token.replace(",", "").replace(".", "")
    else:
        decimal_index = token.rfind(decimal_separator)
        integer_part = token[:decimal_index].replace(",", "").replace(".", "")
        fractional_part = token[decimal_index + 1 :]
        normalized = f"{integer_part}.{fractional_part}"

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def parse_data(raw: str) -> date | None:
    """Parse common receipt issue dates, optionally followed by a timestamp.

    Two-digit years in hyphenated dates default to YY-MM-DD. Four-digit years
    accept YYYY-MM-DD or DD-MM-YYYY. Slashes and dots accept YYYY/MM/DD or DD/MM/YYYY
    (and two-digit year equivalents).

    Args:
        raw: Text containing a candidate date.

    Returns:
        Parsed date object or None when no supported valid date is found.
    """
    text = limpar_texto(raw)
    text = re.sub(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)", "", text)
    text = limpar_texto(text)
    date_patterns = (
        (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", "ymd"),
        (r"(?<!\d)(\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", "ymd"),
        (r"(?<!\d)(\d{8})(?!\d)", "compact"),
        (r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)", "dmy"),
        (r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](\d{2})(?!\d)", "dmy"),
    )

    for pattern, order in date_patterns:
        for match in re.finditer(pattern, text):
            if order == "compact":
                compact = match.group(1)
                year, month, day = int(compact[:4]), int(compact[4:6]), int(compact[6:])
            elif order == "ymd":
                first, second, third = (int(part) for part in match.groups())
                year, month, day = first, second, third
                if year < 100:
                    year += 2000
            else:
                first, second, third = (int(part) for part in match.groups())
                day, month, year = first, second, third
                if year < 100:
                    year += 2000

            try:
                return date(year, month, day)
            except ValueError:
                continue

    return None


def parse_nif(raw: str) -> str | None:
    """Extract a 9-digit Portuguese NIF sequence from text.

    Args:
        raw: Text that may contain a NIF code, with optional spacing.

    Returns:
        The 9-digit string, or None if no 9-digit sequence is found.
    """
    match = re.search(r"(?<!\d)\d(?:[ \t]*\d){8}(?!\d)", raw)
    if match is None:
        return None
    return "".join(match.group().split())


def validar_nif(nif: str) -> bool:
    """Validate allowed leading digit and modulo-11 check digit of a Portuguese NIF.

    Args:
        nif: Candidate 9-digit NIF string.

    Returns:
        True if the NIF has 9 digits and satisfies modulo-11 check digit verification.
    """
    if len(nif) != 9 or not nif.isdigit() or nif[0] not in VALID_NIF_FIRST_DIGITS:
        return False

    checksum = sum(int(digit) * weight for digit, weight in zip(nif[:8], range(9, 1, -1), strict=True))
    check_digit = 11 - checksum % 11
    if check_digit >= 10:
        check_digit = 0
    return int(nif[-1]) == check_digit


def normalizar_entidade(raw: str) -> str:
    """Create a canonical merchant name ignoring accents, punctuation and suffixes.

    Args:
        raw: Merchant name as read from a receipt.

    Returns:
        Lower-cased name without accents, punctuation, or recognized corporate suffixes.
    """
    decomposed = unicodedata.normalize("NFD", raw.lower())
    entity_suffixes = {"lda", "sa", "unipessoal", "sociedade"}
    without_accents = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    without_periods = without_accents.replace(".", "")
    words = "".join(
        character if character.isalnum() or character.isspace() else " " for character in without_periods
    ).split()
    while len(words) > 1 and words[-1] in entity_suffixes:
        words.pop()
    return " ".join(words)


def valores_equivalentes(campo: str, esperado: str | None, obtido: str | None) -> bool:
    """Compare two values using the canonical normalisation rule for their field.

    Args:
        campo: Canonical field name ('total', 'data', 'nif_*', or 'entidade').
        esperado: Reference expected value.
        obtido: Extracted candidate value.

    Returns:
        True if both values are semantically equivalent under the field's rule.

    Raises:
        ValueError: If campo is not a recognized canonical field name.
    """
    esperado = esperado.strip() or None if esperado is not None else None
    obtido = obtido.strip() or None if obtido is not None else None

    if esperado is None or obtido is None:
        return esperado is None and obtido is None

    if campo == "total":
        expected_value = parse_moeda(esperado)
        obtained_value = parse_moeda(obtido)
    elif campo == "data":
        expected_value = parse_data(esperado)
        obtained_value = parse_data(obtido)
    elif campo in {"nif_comerciante", "nif_cliente"}:
        expected_value = parse_nif(esperado)
        obtained_value = parse_nif(obtido)
    elif campo == "entidade":
        expected_value = normalizar_entidade(esperado)
        obtained_value = normalizar_entidade(obtido)
    else:
        raise ValueError(f"Unsupported field: {campo}")

    return expected_value is not None and obtained_value is not None and expected_value == obtained_value
