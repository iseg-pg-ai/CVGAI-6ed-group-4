"""Tests for corpus statistics."""

from datetime import date
from decimal import Decimal
from typing import Literal

from src.dataset_stats import imprimir_resumo, resumo
from src.types import GroundTruthRecord


def _record(
    filename: str,
    grupo: Literal["plano", "amarrotado"],
    entidade: str | None,
    *,
    nif_comerciante: str | None = None,
    nif_cliente: str | None = None,
    data: date | None = None,
    total: Decimal | None = None,
) -> GroundTruthRecord:
    """Build one ground-truth record for statistics tests."""
    return GroundTruthRecord(
        filename=filename,
        grupo=grupo,
        entidade=entidade,
        nif_comerciante=nif_comerciante,
        nif_cliente=nif_cliente,
        data=data,
        total=total,
    )


def test_resumo_counts_groups_merchants_and_present_fields() -> None:
    records = [
        _record(
            "flat-1.jpg",
            "plano",
            "Pingo Doce",
            nif_comerciante="500100144",
            data=date(2026, 10, 24),
            total=Decimal("16.10"),
        ),
        _record(
            "flat-2.jpg",
            "plano",
            "Pingo Doce",
            nif_comerciante="500100144",
            nif_cliente="123456789",
            data=date(2026, 10, 25),
            total=Decimal("9.50"),
        ),
        _record(
            "crumpled-1.jpg",
            "amarrotado",
            "Continente",
            nif_comerciante="501000000",
            nif_cliente="987654321",
            data=date(2026, 10, 26),
            total=Decimal("12.00"),
        ),
        _record("crumpled-2.jpg", "amarrotado", None),
    ]

    result = resumo(records)

    assert result == {
        "total_registos": 4,
        "por_grupo": {
            "plano": {
                "n": 2,
                "entidade": 2,
                "nif_comerciante": 2,
                "nif_cliente": 1,
                "data": 2,
                "total": 2,
            },
            "amarrotado": {
                "n": 2,
                "entidade": 1,
                "nif_comerciante": 1,
                "nif_cliente": 1,
                "data": 1,
                "total": 1,
            },
        },
        "por_comerciante": {
            "(sem entidade)": 1,
            "Continente": 1,
            "Pingo Doce": 2,
        },
        "presenca_campos": {
            "entidade": 3,
            "nif_comerciante": 3,
            "nif_cliente": 2,
            "data": 3,
            "total": 3,
        },
    }


def test_resumo_includes_both_groups_for_empty_corpus() -> None:
    assert resumo([]) == {
        "total_registos": 0,
        "por_grupo": {
            "plano": {"n": 0, "entidade": 0, "nif_comerciante": 0, "nif_cliente": 0, "data": 0, "total": 0},
            "amarrotado": {
                "n": 0,
                "entidade": 0,
                "nif_comerciante": 0,
                "nif_cliente": 0,
                "data": 0,
                "total": 0,
            },
        },
        "por_comerciante": {},
        "presenca_campos": {"entidade": 0, "nif_comerciante": 0, "nif_cliente": 0, "data": 0, "total": 0},
    }


def test_imprimir_resumo_formats_groups_and_merchants(capsys) -> None:
    records = [
        _record("flat.jpg", "plano", "Loja", nif_comerciante="500100144"),
        _record("crumpled.jpg", "amarrotado", None),
    ]

    imprimir_resumo(records)

    output = capsys.readouterr().out
    assert "Grupo" in output
    assert "nif_com" in output
    assert "nif_cli" in output
    assert "plano" in output
    assert "amarrotado" in output
    assert "Loja: 1" in output
    assert "(sem entidade): 1" in output
