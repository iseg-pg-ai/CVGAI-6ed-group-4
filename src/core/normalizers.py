"""Functions to normalise and compare receipt field values.

Ref: Docs/Planning/implementation_plan.md, section 4.3.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation

from src.core.types import EARLIEST_PLAUSIBLE_DATE, FieldName

VALID_NIF_FIRST_DIGITS = frozenset("12356789")

ENTITY_SUFFIXES = frozenset({"lda", "sa", "unipessoal", "unip", "sociedade", "limitada", "s"})
MIN_COMPARABLE_LENGTH = 4
ENTITY_SIMILARITY_THRESHOLD = 0.72


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
    match = re.search(r"(?<!\d)[+-]?\d+(?:[ \u00a0\u202f]\d{3}|[.,]\d+)*(?!\d)", limpar_texto(raw))
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


def validar_data(parsed: date) -> bool:
    """Validate whether a parsed receipt date falls within the plausible issue window.

    Args:
        parsed: Candidate date object.

    Returns:
        True if the date is between EARLIEST_PLAUSIBLE_DATE (2000-01-01) and today inclusive.
    """
    return EARLIEST_PLAUSIBLE_DATE <= parsed <= date.today()


def parse_nif(raw: str) -> str | None:
    """Extract a 9-digit Portuguese NIF sequence from text.

    Args:
        raw: Text that may contain a NIF code, with optional spacing.

    Returns:
        The 9-digit string, or None if no 9-digit sequence is found.
    """
    cleaned = re.sub(r"(?i)\bP[T71Il]\s*(?=\d)", "PT ", raw)

    match = re.search(r"(?<!\d)\d(?:[ \t]*\d){8}(?!\d)", cleaned)
    if match is not None:
        return "".join(match.group().split())

    match10 = re.search(r"(?<!\d)7(\d{9})(?!\d)", cleaned)
    if match10 is not None and validar_nif(match10.group(1)):
        return match10.group(1)

    slash_match = re.search(r"(?<!\d)(\d+/\d+)(?!\d)", cleaned)
    if slash_match is not None and len(slash_match.group(1)) == 9:
        for replacement in ("7", "1"):
            candidate = slash_match.group(1).replace("/", replacement)
            if validar_nif(candidate):
                return candidate

    return None


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
    without_accents = "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    without_periods = without_accents.replace(".", "")
    words = "".join(
        character if character.isalnum() or character.isspace() else " " for character in without_periods
    ).split()
    while len(words) > 1 and words[-1] in ENTITY_SUFFIXES:
        words.pop()
    return " ".join(words)


def _optional_text(value: str | None) -> str | None:
    """Normalise an optional raw value, treating blanks as absent.

    Args:
        value: Raw field text or None.

    Returns:
        Stripped text, or None when the input was absent or whitespace only.
    """
    if value is None:
        return None
    return value.strip() or None


def _parsed_equal(expected: object, obtained: object) -> bool:
    """Return True only when both parsed values exist and are equal."""
    return expected is not None and obtained is not None and expected == obtained


def _contains_words(haystack: str, needle: str) -> bool:
    """Return True when needle appears in haystack on token boundaries."""
    return f" {needle} " in f" {haystack} "


def _similar_enough(left: str, right: str) -> bool:
    """Return True when the similarity ratio reaches ENTITY_SIMILARITY_THRESHOLD."""
    matcher = difflib.SequenceMatcher(None, left, right)
    if matcher.quick_ratio() < ENTITY_SIMILARITY_THRESHOLD:
        return False
    return matcher.ratio() >= ENTITY_SIMILARITY_THRESHOLD


def _entidades_equivalentes(expected: str, obtained: str) -> bool:
    """Compare merchant names after canonical normalisation, tolerating OCR noise and filler words."""
    canonical_expected = normalizar_entidade(expected)
    canonical_obtained = normalizar_entidade(obtained)
    if not canonical_expected or not canonical_obtained:
        return False
    if canonical_expected == canonical_obtained:
        return True
    if len(canonical_expected) >= MIN_COMPARABLE_LENGTH and _contains_words(canonical_obtained, canonical_expected):
        return True
    if len(canonical_obtained) >= MIN_COMPARABLE_LENGTH and _contains_words(canonical_expected, canonical_obtained):
        return True
    if len(canonical_expected) < MIN_COMPARABLE_LENGTH or len(canonical_obtained) < MIN_COMPARABLE_LENGTH:
        return False
    if _similar_enough(canonical_expected, canonical_obtained):
        return True
    expected_words = canonical_expected.split()
    obtained_words = canonical_obtained.split()
    width = len(expected_words)
    return any(
        _similar_enough(canonical_expected, " ".join(obtained_words[index : index + width]))
        for index in range(len(obtained_words) - width + 1)
    )


def _comparar_total(expected: str, obtained: str) -> bool:
    """Compare two monetary values as parsed Decimals."""
    return _parsed_equal(parse_moeda(expected), parse_moeda(obtained))


def _comparar_data(expected: str, obtained: str) -> bool:
    """Compare two dates parsed from free text."""
    return _parsed_equal(parse_data(expected), parse_data(obtained))


def _comparar_nif(expected: str, obtained: str) -> bool:
    """Compare two NIFs after nine-digit extraction."""
    return _parsed_equal(parse_nif(expected), parse_nif(obtained))


Comparador = Callable[[str, str], bool]

_COMPARADORES: dict[FieldName, Comparador] = {
    "total": _comparar_total,
    "data": _comparar_data,
    "nif_comerciante": _comparar_nif,
    "nif_cliente": _comparar_nif,
    "entidade": _entidades_equivalentes,
}


def valores_equivalentes(campo: FieldName, esperado: str | None, obtido: str | None) -> bool:
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
    comparador = _COMPARADORES.get(campo)
    if comparador is None:
        raise ValueError(f"Unsupported field: {campo}")

    expected = _optional_text(esperado)
    obtained = _optional_text(obtido)
    if expected is None or obtained is None:
        return expected is None and obtained is None

    return comparador(expected, obtained)
