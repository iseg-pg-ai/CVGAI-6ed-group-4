"""Tests for field scoring, metric aggregation, and report tables."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from src.scoring import agregar, escrever_tabelas, pontuar_registo
from src.types import Extraction, FieldValue, GroundTruthRecord, MatchResult


def _record(*, customer_nif: str | None = None) -> GroundTruthRecord:
    """Build a representative validated ground-truth row."""
    return GroundTruthRecord(
        filename="receipt.jpg",
        grupo="direito",
        entidade="Pingo Doce",
        nif_comerciante="500100144",
        nif_cliente=customer_nif,
        data=date(2026, 10, 24),
        total=Decimal("16.10"),
    )


def _extraction(
    *,
    entidade: str | None = "Pingo Doce",
    nif_comerciante: str | None = "500100144",
    nif_cliente: str | None = None,
    data: str | None = "2026-10-24",
    total: str | None = "16.10",
) -> Extraction:
    """Build extraction data from optional values."""
    return Extraction(
        entidade=FieldValue(value=entidade, status="ENCONTRADO" if entidade is not None else "AUSENTE"),
        nif_comerciante=FieldValue(
            value=nif_comerciante, status="ENCONTRADO" if nif_comerciante is not None else "AUSENTE"
        ),
        nif_cliente=FieldValue(value=nif_cliente, status="ENCONTRADO" if nif_cliente is not None else "AUSENTE"),
        data=FieldValue(value=data, status="ENCONTRADO" if data is not None else "AUSENTE"),
        total=FieldValue(value=total, status="ENCONTRADO" if total is not None else "AUSENTE"),
        method="regex",
        elapsed_ms=0.0,
    )


def test_pontuar_registo_compares_all_fields_in_canonical_order() -> None:
    """A record produces five matches and normalizes typed date/decimal reference values."""
    results = pontuar_registo(
        _record(customer_nif="123456789"),
        _extraction(
            entidade="Pingo Doce, Lda.",
            nif_cliente="123456789",
        ),
    )

    assert [result.campo for result in results] == [
        "entidade",
        "nif_comerciante",
        "nif_cliente",
        "data",
        "total",
    ]
    assert all(result.acertou for result in results)
    assert results[3].esperado == "2026-10-24"
    assert results[4].esperado == "16.10"


def test_pontuar_registo_marks_false_positive_and_omission_separately() -> None:
    """An invented customer NIF on a receipt without one is a false positive."""
    results = pontuar_registo(
        _record(),
        _extraction(nif_comerciante="500100145", nif_cliente="123456789", data=None),
    )

    by_field = {result.campo: result for result in results}
    assert by_field["nif_cliente"].esperado is None
    assert by_field["nif_cliente"].falso_positivo
    assert not by_field["nif_cliente"].omissao
    assert not by_field["nif_cliente"].acertou

    assert by_field["nif_comerciante"].esperado == "500100144"
    assert by_field["nif_comerciante"].obtido == "500100145"
    assert not by_field["nif_comerciante"].falso_positivo
    assert not by_field["nif_comerciante"].omissao
    assert not by_field["nif_comerciante"].acertou

    assert by_field["data"].esperado == "2026-10-24"
    assert by_field["data"].obtido is None
    assert not by_field["data"].falso_positivo
    assert by_field["data"].omissao
    assert not by_field["data"].acertou


def test_pontuar_registo_rejects_invalido_status() -> None:
    """A field marked INVALIDO never scores acertou=True when an expected value exists."""
    from src.types import FieldValue

    ext = _extraction()
    ext = Extraction(
        entidade=ext.entidade,
        nif_comerciante=FieldValue(value="500100144", status="INVALIDO"),
        nif_cliente=ext.nif_cliente,
        data=ext.data,
        total=ext.total,
        method="regex",
        elapsed_ms=1.0,
    )
    results = pontuar_registo(_record(), ext)
    by_field = {result.campo: result for result in results}
    assert by_field["nif_comerciante"].acertou is False


def test_agregar_uses_the_defined_denominators_for_each_rate() -> None:
    """Accuracy uses all fields; false positives/omissions use absent/present truths."""
    results = pontuar_registo(
        _record(),
        _extraction(nif_comerciante="500100145", nif_cliente="123456789", data=None),
    )

    metrics = agregar(results)

    assert metrics["global"] == {
        "taxa_acerto": 0.4,
        "taxa_falso_positivo": 1.0,
        "taxa_omissao": 0.25,
    }
    assert metrics["por_campo"]["nif_cliente"] == {
        "taxa_acerto": 0.0,
        "taxa_falso_positivo": 1.0,
        "taxa_omissao": 0.0,
    }
    assert metrics["por_grupo"]["direito"] == metrics["global"]
    assert metrics["por_grupo"]["amarrotado"]["taxa_acerto"] == 0.0


def test_absent_truth_and_absent_prediction_count_as_correct() -> None:
    """AUSENTE is correct when the reference field is also absent."""
    customer_result = pontuar_registo(_record(), _extraction())[2]

    assert customer_result.acertou
    assert not customer_result.falso_positivo
    assert not customer_result.omissao
    assert agregar([customer_result])["global"]["taxa_acerto"] == 1.0


def test_agregar_returns_zero_rates_when_denominators_are_empty() -> None:
    """Empty subsets are represented with zero rates instead of division failures."""
    metrics = agregar([])

    assert metrics["global"] == {
        "taxa_acerto": 0.0,
        "taxa_falso_positivo": 0.0,
        "taxa_omissao": 0.0,
    }
    assert metrics["por_campo"]["total"] == metrics["global"]


def test_escrever_tabelas_creates_csv_details_and_markdown_summary(tmp_path: Path) -> None:
    """The report writer creates both requested files under the destination directory."""
    results = pontuar_registo(_record(), _extraction(nif_cliente="123456789", data=None))
    destination = tmp_path / "report"

    escrever_tabelas(results, destination)

    with (destination / "metrics.csv").open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    customer_row = next(row for row in rows if row["campo"] == "nif_cliente")
    assert customer_row["esperado"] == ""
    assert customer_row["obtido"] == "123456789"
    assert customer_row["falso_positivo"] == "true"
    assert customer_row["omissao"] == "false"

    markdown = (destination / "metrics.md").read_text(encoding="utf-8")
    assert "## Global" in markdown
    assert "## Por campo" in markdown
    assert "## Por grupo" in markdown
    assert "taxa_falso_positivo" in markdown


def test_escrever_tabelas_handles_empty_results(tmp_path: Path) -> None:
    """Empty inputs still produce headers and a zero-valued summary."""
    escrever_tabelas([], tmp_path)

    with (tmp_path / "metrics.csv").open(encoding="utf-8-sig", newline="") as csv_file:
        assert list(csv.reader(csv_file)) == [
            ["filename", "grupo", "campo", "esperado", "obtido", "acertou", "falso_positivo", "omissao"]
        ]
    assert "0.0000" in (tmp_path / "metrics.md").read_text(encoding="utf-8")


def test_agregar_accepts_empty_group_and_field_values_in_input() -> None:
    """Additional keys in the shared MatchResult contract are retained in aggregates."""
    custom_result = MatchResult(
        filename="other.jpg",
        grupo="custom",  # type: ignore[arg-type]
        campo="custom_field",  # type: ignore[arg-type]
        esperado=None,
        obtido=None,
        acertou=True,
        falso_positivo=False,
        omissao=False,
    )

    metrics = agregar([custom_result])

    assert metrics["por_grupo"]["custom"]["taxa_acerto"] == 1.0
    assert metrics["por_campo"]["custom_field"]["taxa_acerto"] == 1.0
