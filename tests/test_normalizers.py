"""Unit and acceptance tests for pure field normalisers and comparators.

Encodes the exact acceptance table from Docs/Planning/implementation_plan.md (section 6, WP-02)
along with regression cases for edge cases discovered during review.

Ref: Docs/Planning/implementation_plan.md, section 6, WP-02.
"""

from datetime import date
from decimal import Decimal

import pytest

from src.normalizers import (
    limpar_texto,
    normalizar_entidade,
    parse_data,
    parse_moeda,
    parse_nif,
    validar_nif,
    valores_equivalentes,
)

# ---------------------------------------------------------------------------
# Acceptance table from implementation_plan.md, section 6 (WP-02)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("16,10 EUR", Decimal("16.10")),
        ("11.57", Decimal("11.57")),
        (" 1 234,56 ", Decimal("1234.56")),
        ("TOTAL", None),
    ],
)
def test_acceptance_parse_moeda(raw: str, expected: Decimal | None) -> None:
    """Verifies the four currency cases from the specification acceptance table."""
    result = parse_moeda(raw)
    assert result == expected, f"parse_moeda({raw!r}) returned {result!r}, expected {expected!r}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("26-10-27 13:45:42", date(2026, 10, 27)),
        ("20261027", date(2026, 10, 27)),
        ("31-02-26", None),
    ],
)
def test_acceptance_parse_data(raw: str, expected: date | None) -> None:
    """Verifies the three date cases from the specification acceptance table."""
    result = parse_data(raw)
    assert result == expected, f"parse_data({raw!r}) returned {result!r}, expected {expected!r}"


def test_acceptance_nif() -> None:
    """Verifies the three NIF cases from the specification acceptance table."""
    extracted = parse_nif("NIF: 518 396 894")
    assert extracted == "518396894", f"parse_nif extracted {extracted!r}, expected '518396894'"
    assert validar_nif("518396894") is True, "518396894 must be valid under modulo-11 checksum"
    assert validar_nif("518396895") is False, "518396895 must be rejected (invalid checksum)"


def test_acceptance_normalizar_entidade() -> None:
    """Verifies the merchant normalisation case from the specification acceptance table."""
    normalized = normalizar_entidade("O DAS BIFANAS, LDA.")
    assert normalized == "o das bifanas", f"got {normalized!r}, expected 'o das bifanas'"


@pytest.mark.parametrize(
    ("campo", "esperado", "obtido", "expected_match"),
    [
        ("total", "16,10", "16.1", True),
        ("entidade", None, None, True),
        ("entidade", "Pingo Doce", "Continente", False),
    ],
)
def test_acceptance_valores_equivalentes(
    campo: str, esperado: str | None, obtido: str | None, expected_match: bool
) -> None:
    """Verifies the three equivalence cases from the specification acceptance table."""
    result = valores_equivalentes(campo, esperado, obtido)
    assert result is expected_match, (
        f"valores_equivalentes({campo!r}, {esperado!r}, {obtido!r}) returned {result}, expected {expected_match}"
    )


# ---------------------------------------------------------------------------
# Additional coverage and boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Loja\t  teste\n", "Loja teste"),
        ("A\u0000B", "AB"),
        ("", ""),
        ("Palavra com \u00a0 espaço não separável\u202fe fino", "Palavra com espaço não separável e fino"),
    ],
)
def test_limpar_texto(raw: str, expected: str) -> None:
    """Strips control chars, unifies unicode spaces, collapses internal whitespace."""
    assert limpar_texto(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total: 1.234,56 EUR", Decimal("1234.56")),
        ("Total: 1 234,56 EUR", Decimal("1234.56")),
        ("Total: 12.50 EUR", Decimal("12.50")),
        ("5,86€", Decimal("5.86")),
        ("Sem valor", None),
        ("1234", Decimal("1234")),
        ("Total: 0,50 €", Decimal("0.50")),
    ],
)
def test_parse_moeda_additional(raw: str, expected: Decimal | None) -> None:
    """Tests integers without separators, European/US decimals, and euro symbol suffixes."""
    assert parse_moeda(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Emitido em 2024-02-29 12:30", date(2024, 2, 29)),
        ("Emitido em 29/02/24", date(2024, 2, 29)),
        ("Emitido em 2026/10/24", date(2026, 10, 24)),
        ("Emitido em 2026.10.24", date(2026, 10, 24)),
        ("Data: 24/10/2026", date(2026, 10, 24)),
        ("Data: 24.10.2026", date(2026, 10, 24)),
        ("Hora: 14:30 Data: 2026-10-24", date(2026, 10, 24)),
        ("Data: 31/02/2024", None),
    ],
)
def test_parse_data_additional(raw: str, expected: date | None) -> None:
    """Tests four-digit years with slash/dot, leap years, and time preceding date."""
    assert parse_data(raw) == expected


def test_validar_nif_legal_first_digits() -> None:
    """Portuguese NIFs starting with 1, 2, 3, 5, 6, 7, 8, 9 must all be accepted when checksum matches.

    Prefix 7 denotes 'herança indivisa' and is common on legal and customer receipts.
    NIF 500000000 tests check digit 0 (11 - 11 = 0).
    """
    assert validar_nif("700000003") is True, "NIF starting with 7 (herança indivisa) must be allowed"
    assert validar_nif("500000000") is True, "NIF resulting in check digit 0 must validate correctly"
    assert validar_nif("501964843") is True
    assert validar_nif("501964844") is False, "Wrong check digit must fail"
    assert validar_nif("001964843") is False, "Prefix 0 is disallowed for Portuguese NIFs"
    assert validar_nif("401964843") is False, "Prefix 4 is disallowed for Portuguese NIFs"
    assert validar_nif("12345678") is False, "Must have exactly 9 digits"
    assert validar_nif("1234567890") is False, "Must have exactly 9 digits"
    assert validar_nif("50196484A") is False, "Must contain digits only"


def test_parse_nif_additional() -> None:
    """Extracts nine-digit sequence across dots, tabs and common NIF prefixes."""
    assert parse_nif("NIF: 501 964 843") == "501964843"
    assert parse_nif("Contribuinte: 518396894") == "518396894"
    assert parse_nif("NIF inválido: 1234567890") is None


def test_normalizar_entidade_legal_suffixes() -> None:
    """Dotted and spaced legal suffixes (S.A., Lda., Unipessoal Lda) must be stripped symmetrically."""
    assert normalizar_entidade("  Café Bom, Lda. ") == "cafe bom"
    assert normalizar_entidade("Empresa Unipessoal, Lda.") == "empresa"
    assert normalizar_entidade("Pingo Doce, S.A.") == "pingo doce", (
        "Dotted 'S.A.' must collapse to 'sa' and be removed like 'SA'"
    )
    assert normalizar_entidade("Pingo Doce SA") == "pingo doce"
    assert normalizar_entidade("X, L.da") == "x", "Dotted 'L.da' must be removed like 'Lda'"


@pytest.mark.parametrize(
    ("campo", "esperado", "obtido", "equivalente"),
    [
        ("total", "1.234,56 €", "1234.56", True),
        ("data", "29/02/24", "2024-02-29", True),
        ("nif_comerciante", "501 964 843", "501964843", True),
        ("entidade", "Café Bom, Lda.", "Cafe Bom", True),
        ("entidade", "Pingo Doce, S.A.", "Pingo Doce SA", True),
        ("total", "1,00", "2,00", False),
        ("total", None, None, True),
        ("total", None, "1,00", False),
        ("total", "1,00", None, False),
        ("nif_cliente", "", "", True),
        ("nif_cliente", None, "", True),
        ("nif_cliente", "", None, True),
    ],
)
def test_valores_equivalentes_additional(
    campo: str, esperado: str | None, obtido: str | None, equivalente: bool
) -> None:
    """Verifies symmetric handling of blank strings, None, and dotted suffixes."""
    assert valores_equivalentes(campo, esperado, obtido) is equivalente


def test_valores_equivalentes_rejeita_campo_desconhecido() -> None:
    """Raises ValueError when given an unsupported canonical field name."""
    with pytest.raises(ValueError, match="Unsupported field"):
        valores_equivalentes("desconhecido", "a", "a")
