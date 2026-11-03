"""Load and validate the receipt dataset ground truth.

Ref: Docs/Planning/implementation_plan.md, section 6, WP-03.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.normalizers import validar_nif
from src.settings import Settings, get_settings
from src.types import (
    CONSUMIDOR_FINAL_NIF,
    EARLIEST_PLAUSIBLE_DATE,
    FIELD_NAMES,
    GROUPS,
    GroundTruthRecord,
    GroupName,
)

_CSV_COLUMNS = (
    "filename",
    "grupo",
    *FIELD_NAMES,
)


@dataclass(frozen=True, slots=True)
class GoldenSetError(Exception):
    """Aggregate validation failures found in a golden-set CSV.

    Attributes:
        errors: Human-readable problems, including their CSV row when applicable.
    """

    errors: tuple[str, ...]

    def __str__(self) -> str:
        """Return all validation failures as one readable message."""
        return "Golden set validation failed:\n" + "\n".join(f"- {error}" for error in self.errors)


def _read_rows(path: Path) -> tuple[list[tuple[int, list[str]]], list[str]]:
    """Read CSV records while preserving blank rows and reporting I/O/CSV failures."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.reader(csv_file)
            rows = [(row_number, row) for row_number, row in enumerate(reader, start=1)]
    except (OSError, UnicodeError, csv.Error) as error:
        return [], [f"Não foi possível ler o CSV '{path}': {error}"]

    if not rows:
        return [], [f"CSV '{path}' vazio; cabeçalho obrigatório em falta."]

    header = rows[0][1]
    if tuple(header) != _CSV_COLUMNS:
        expected = ", ".join(_CSV_COLUMNS)
        actual = ", ".join(header) if header else "(vazio)"
        return [], [f"Cabeçalho inválido: esperado [{expected}], recebido [{actual}]."]

    return rows[1:], []


def _validate_csv(path: Path, receipts_dir: Path) -> tuple[list[GroundTruthRecord], list[str]]:
    """Validate all rows and return records alongside every discovered problem."""
    rows, errors = _read_rows(path)
    if errors:
        return [], errors

    records: list[GroundTruthRecord] = []
    seen_filenames: set[str] = set()
    today = date.today()

    for row_number, values in rows:
        prefix = f"Linha {row_number}"
        if not values or not any(value.strip() for value in values):
            errors.append(f"{prefix}: linha totalmente vazia.")
            continue
        if len(values) > len(_CSV_COLUMNS):
            errors.append(f"{prefix}: coluna extra (esperadas {len(_CSV_COLUMNS)}, recebidas {len(values)}).")
        elif len(values) < len(_CSV_COLUMNS):
            errors.append(f"{prefix}: colunas em falta (esperadas {len(_CSV_COLUMNS)}, recebidas {len(values)}).")
        fields = [value.strip() for value in values[: len(_CSV_COLUMNS)]]
        fields.extend([""] * (len(_CSV_COLUMNS) - len(fields)))
        filename, group, entity, merchant_nif, customer_nif, raw_date, raw_total = fields

        row_errors: list[str] = []
        filename_is_safe = False
        if not filename:
            row_errors.append("filename vazio")
        else:
            filename_is_safe = (
                not Path(filename).is_absolute()
                and Path(filename).name == filename
                and "/" not in filename
                and "\\" not in filename
                and ":" not in filename
                and "\0" not in filename
                and filename not in {".", ".."}
            )
            if not filename_is_safe:
                row_errors.append(f"filename deve ser um nome de ficheiro simples: {filename!r}")
            if filename in seen_filenames:
                row_errors.append(f"filename duplicado: {filename!r}")
            seen_filenames.add(filename)
            if filename_is_safe and not (receipts_dir / filename).is_file():
                row_errors.append(f"ficheiro não encontrado em '{receipts_dir}': {filename!r}")

        group_value: GroupName | None = None
        if group in GROUPS:
            group_value = group  # type: ignore[assignment]
        else:
            row_errors.append(f"grupo inválido: {group!r}")

        for field_name, nif in (
            ("nif_comerciante", merchant_nif),
            ("nif_cliente", customer_nif),
        ):
            if nif:
                if nif == CONSUMIDOR_FINAL_NIF:
                    row_errors.append(
                        f"{field_name} não deve usar o sentinela de consumidor final {CONSUMIDOR_FINAL_NIF!r}; "
                        f"deixe a célula em branco se ausente"
                    )
                elif re.fullmatch(r"[0-9]{9}", nif) is None or not validar_nif(nif):
                    row_errors.append(f"{field_name} inválido: {nif!r}")

        parsed_date: date | None = None
        if raw_date:
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw_date) is None:
                row_errors.append(f"data inválida (formato esperado AAAA-MM-DD): {raw_date!r}")
            else:
                try:
                    parsed_date = date.fromisoformat(raw_date)
                except ValueError:
                    row_errors.append(f"data inválida: {raw_date!r}")
                else:
                    if parsed_date > today:
                        row_errors.append(f"data no futuro: {raw_date!r}")
                    elif parsed_date < EARLIEST_PLAUSIBLE_DATE:
                        row_errors.append(f"data anterior a {EARLIEST_PLAUSIBLE_DATE.isoformat()}: {raw_date!r}")

        parsed_total: Decimal | None = None
        if raw_total:
            if re.fullmatch(r"\d+(?:\.\d+)?", raw_total) is None:
                row_errors.append(f"total inválido (formato decimal com ponto esperado): {raw_total!r}")
            else:
                try:
                    parsed_total = Decimal(raw_total)
                except InvalidOperation:
                    row_errors.append(f"total inválido: {raw_total!r}")
                else:
                    if not parsed_total.is_finite() or parsed_total <= 0:
                        row_errors.append(f"total deve ser maior que zero: {raw_total!r}")

        errors.extend(f"{prefix}: {error}." for error in row_errors)
        if not row_errors and group_value is not None:
            records.append(
                GroundTruthRecord(
                    filename=filename,
                    grupo=group_value,
                    entidade=entity or None,
                    nif_comerciante=merchant_nif or None,
                    nif_cliente=customer_nif or None,
                    data=parsed_date,
                    total=parsed_total,
                )
            )

    return records, errors


def carregar_ground_truth(path: Path, receipts_dir: Path | None = None) -> list[GroundTruthRecord]:
    """Read and validate the CSV. Raise GoldenSetError with every failure found."""
    effective_receipts_dir = receipts_dir if receipts_dir is not None else get_settings().receipts_dir
    records, errors = _validate_csv(path, effective_receipts_dir)
    if errors:
        raise GoldenSetError(tuple(errors))
    return records


def validar_ground_truth(path: Path, receipts_dir: Path) -> list[str]:
    """Return the list of problems found; an empty list means valid."""

    _, errors = _validate_csv(path, receipts_dir)
    return errors


def main(argv: Sequence[str] | None = None, settings_override: Settings | None = None) -> None:
    """Aggregate every validation error into one readable message."""

    settings = settings_override or get_settings()
    parser = argparse.ArgumentParser(description="Validate the receipt golden-set CSV")
    parser.add_argument("--csv", type=Path, default=settings.ground_truth_path, help="Ground-truth CSV path")
    parser.add_argument("--receipts-dir", type=Path, default=settings.receipts_dir, help="Receipt image directory")
    args = parser.parse_args(argv)

    problems = validar_ground_truth(args.csv, args.receipts_dir)
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit(1)
    print(f"Golden set válido: {args.csv}")


if __name__ == "__main__":
    main()
