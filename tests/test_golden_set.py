"""Tests for reading and validating the receipt golden set."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src.golden_set import GoldenSetError, carregar_ground_truth, validar_ground_truth
from src.settings import get_settings
from src.types import GroundTruthRecord

CSV_COLUMNS = ["filename", "grupo", "entidade", "nif_comerciante", "nif_cliente", "data", "total"]
VALID_ROW = ["receipt.jpg", "direito", "Pingo Doce", "500100144", "123456789", "2026-10-24", "16.10"]


def write_csv(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    """Write test CSV data without changing the repository."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header if header is not None else CSV_COLUMNS)
        writer.writerows(rows)


def test_repository_ground_truth_is_valid(repo_root: Path) -> None:
    """The checked-in header-only ground-truth file is currently a valid empty set."""
    assert validar_ground_truth(repo_root / "data" / "ground_truth.csv", repo_root / "data" / "receipts") == []


def test_valid_rows_are_converted_to_typed_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader converts optional cells and scalar values into the shared data contract."""
    receipts_dir = tmp_path / "data" / "receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / VALID_ROW[0]).touch()
    csv_path = tmp_path / "data" / "ground_truth.csv"
    write_csv(csv_path, [VALID_ROW])
    configured = replace(get_settings(), receipts_dir=receipts_dir)
    monkeypatch.setattr("src.golden_set.get_settings", lambda: configured)

    records = carregar_ground_truth(csv_path)

    assert records == [
        GroundTruthRecord(
            filename="receipt.jpg",
            grupo="direito",
            entidade="Pingo Doce",
            nif_comerciante="500100144",
            nif_cliente="123456789",
            data=date(2026, 10, 24),
            total=Decimal("16.10"),
        )
    ]


def test_optional_empty_fields_become_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty optional fields represent absence rather than placeholder values."""

    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "receipt.jpg").touch()
    csv_path = tmp_path / "truth.csv"
    write_csv(csv_path, [["receipt.jpg", "amarrotado", "", "", "", "", ""]])
    monkeypatch.setattr("src.golden_set.get_settings", lambda: replace(get_settings(), receipts_dir=receipts_dir))

    record = carregar_ground_truth(csv_path)[0]

    assert record.entidade is None
    assert record.nif_comerciante is None
    assert record.nif_cliente is None
    assert record.data is None
    assert record.total is None


def test_validation_reports_missing_file_invalid_nif_future_date_duplicate_and_missing_column(
    tmp_path: Path,
) -> None:
    """Validation accumulates independent failures instead of stopping at the first one."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    csv_path = tmp_path / "truth.csv"
    invalid_row = ["missing.jpg", "direito", "Loja", "123456780", "", "2999-01-01", "16.10"]
    duplicate_row = ["missing.jpg", "direito", "Loja", "", "", "", "16.10"]
    write_csv(csv_path, [invalid_row, duplicate_row])

    problems = validar_ground_truth(csv_path, receipts_dir)

    assert any("ficheiro não encontrado" in problem for problem in problems)
    assert any("nif_comerciante inválido" in problem for problem in problems)
    assert any("data no futuro" in problem for problem in problems)
    assert any("filename duplicado" in problem for problem in problems)

    short_csv = tmp_path / "short.csv"
    write_csv(short_csv, [VALID_ROW[:-1]])
    short_problems = validar_ground_truth(short_csv, receipts_dir)
    assert any("colunas em falta" in problem for problem in short_problems)


@pytest.mark.parametrize(
    ("row", "expected_error"),
    [
        (["receipt.jpg", "inclinado", "", "", "", "", ""], "grupo inválido"),
        (["receipt.jpg", "direito", "", "500100145", "", "", ""], "nif_comerciante inválido"),
        (["receipt.jpg", "direito", "", "", "123456780", "", ""], "nif_cliente inválido"),
        (["receipt.jpg", "direito", "", "", "", "31-02-2026", ""], "formato esperado AAAA-MM-DD"),
        (["receipt.jpg", "direito", "", "", "", "2026-02-30", ""], "data inválida"),
        (["receipt.jpg", "direito", "", "", "", "", "0"], "maior que zero"),
        (["receipt.jpg", "direito", "", "", "", "", "1,20"], "formato decimal com ponto esperado"),
    ],
)
def test_validation_reports_invalid_field_values(
    tmp_path: Path,
    row: list[str],
    expected_error: str,
) -> None:
    """Each constrained field reports its own validation failure."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "receipt.jpg").touch()
    csv_path = tmp_path / "truth.csv"
    write_csv(csv_path, [row])

    problems = validar_ground_truth(csv_path, receipts_dir)

    assert any(expected_error in problem for problem in problems)


def test_validation_reports_bad_header_blank_row_extra_column_and_unsafe_filename(tmp_path: Path) -> None:
    """Structural and filename validation cover malformed CSV rows."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    csv_path = tmp_path / "truth.csv"
    write_csv(csv_path, [VALID_ROW + ["unexpected"]], header=CSV_COLUMNS[:-1])
    assert any("Cabeçalho inválido" in problem for problem in validar_ground_truth(csv_path, receipts_dir))

    write_csv(csv_path, [VALID_ROW + ["unexpected"]])
    assert any("coluna extra" in problem for problem in validar_ground_truth(csv_path, receipts_dir))

    csv_path.write_text(",".join(CSV_COLUMNS) + "\n\n", encoding="utf-8")
    assert any("linha totalmente vazia" in problem for problem in validar_ground_truth(csv_path, receipts_dir))

    write_csv(csv_path, [["..\\outside.jpg", "direito", "", "", "", "", ""]])
    assert any("nome de ficheiro simples" in problem for problem in validar_ground_truth(csv_path, receipts_dir))


def test_loader_raises_aggregated_golden_set_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader raises one exception containing every issue found."""
    csv_path = tmp_path / "truth.csv"
    write_csv(csv_path, [["", "invalido", "", "", "", "", ""]])
    monkeypatch.setattr(
        "src.golden_set.get_settings",
        lambda: replace(get_settings(), receipts_dir=tmp_path / "receipts"),
    )

    with pytest.raises(GoldenSetError) as error:
        carregar_ground_truth(csv_path)

    assert len(error.value.errors) >= 2
    assert "Golden set validation failed" in str(error.value)
