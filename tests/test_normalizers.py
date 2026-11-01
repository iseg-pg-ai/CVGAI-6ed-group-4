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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Loja\t  teste\n", "Loja teste"),
        ("A\u0000B", "AB"),
        ("", ""),
    ],
)
def test_limpar_texto(raw: str, expected: str) -> None:
    assert limpar_texto(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total: 1.234,56 EUR", Decimal("1234.56")),
        ("Total: 1 234,56 EUR", Decimal("1234.56")),
        ("Total: 12.50 EUR", Decimal("12.50")),
        ("Sem valor", None),
    ],
)
def test_parse_moeda(raw: str, expected: Decimal | None) -> None:
    assert parse_moeda(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Emitido em 2024-02-29 12:30", date(2024, 2, 29)),
        ("Emitido em 29/02/24", date(2024, 2, 29)),
        ("Data: 20240229", date(2024, 2, 29)),
        ("Data: 31/02/2024", None),
    ],
)
def test_parse_data(raw: str, expected: date | None) -> None:
    assert parse_data(raw) == expected


def test_parse_nif_and_validar_nif() -> None:
    nif = "501964843"

    assert parse_nif("NIF: 501 964 843") == nif
    assert validar_nif(nif)
    assert not validar_nif("501964844")
    assert not validar_nif("001964843")
    assert parse_nif("NIF inválido: 1234567890") is None


def test_normalizar_entidade() -> None:
    assert normalizar_entidade("  Café Bom, Lda. ") == "cafe bom"
    assert normalizar_entidade("Empresa Unipessoal, Lda.") == "empresa"


@pytest.mark.parametrize(
    ("campo", "esperado", "obtido", "equivalente"),
    [
        ("total", "1.234,56 €", "1234.56", True),
        ("data", "29/02/24", "2024-02-29", True),
        ("nif_comerciante", "501 964 843", "501964843", True),
        ("entidade", "Café Bom, Lda.", "Cafe Bom", True),
        ("total", "1,00", "2,00", False),
        ("total", None, None, True),
        ("total", None, "1,00", False),
    ],
)
def test_valores_equivalentes(campo: str, esperado: str | None, obtido: str | None, equivalente: bool) -> None:
    assert valores_equivalentes(campo, esperado, obtido) is equivalente


def test_valores_equivalentes_rejeita_campo_desconhecido() -> None:
    with pytest.raises(ValueError, match="Unsupported field"):
        valores_equivalentes("desconhecido", "a", "a")
