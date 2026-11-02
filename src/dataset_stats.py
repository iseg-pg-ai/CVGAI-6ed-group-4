"""Summarize the receipt corpus by image group, merchant, and field presence.

Ref: Docs/Planning/implementation_plan.md, section 6, WP-14.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from src.types import FIELD_NAMES, GROUPS, GroundTruthRecord

if TYPE_CHECKING:
    from src.settings import Settings

_FIELD_LABELS = {
    "entidade": "entidade",
    "nif_comerciante": "nif_com",
    "nif_cliente": "nif_cli",
    "data": "data",
    "total": "total",
}


def resumo(registos: list[GroundTruthRecord]) -> dict[str, object]:
    """Counts by group, by merchant and by presence of each field.

    The returned dictionary contains ``total_registos``, ``por_grupo``,
    ``por_comerciante``, and ``presenca_campos``. Group summaries include ``n``
    and a count for each canonical field. An absent entity is counted under
    ``"(sem entidade)"`` in the merchant summary.

    Args:
        registos: Validated ground-truth records to summarize.

    Returns:
        A summary dictionary. Both known groups are included, even when empty.
    """
    group_counts: dict[str, dict[str, int]] = {group: {"n": 0, **dict.fromkeys(FIELD_NAMES, 0)} for group in GROUPS}
    merchant_counts: Counter[str] = Counter()
    field_counts = dict.fromkeys(FIELD_NAMES, 0)

    for record in registos:
        group_counts[record.grupo]["n"] += 1
        merchant_counts[record.entidade or "(sem entidade)"] += 1
        for field in FIELD_NAMES:
            if getattr(record, field) is not None:
                group_counts[record.grupo][field] += 1
                field_counts[field] += 1

    return {
        "total_registos": len(registos),
        "por_grupo": group_counts,
        "por_comerciante": dict(sorted(merchant_counts.items())),
        "presenca_campos": field_counts,
    }


def imprimir_resumo(registos: list[GroundTruthRecord]) -> None:
    """Format the summary as a text table for the terminal.

    Args:
        registos: Validated ground-truth records to summarize.
    """
    statistics = resumo(registos)
    group_counts = statistics["por_grupo"]
    merchant_counts = statistics["por_comerciante"]
    assert isinstance(group_counts, dict)
    assert isinstance(merchant_counts, dict)

    headers = ("Grupo", "N", *(_FIELD_LABELS[field] for field in FIELD_NAMES))
    rows = [
        (
            group,
            group_counts[group]["n"],
            *(group_counts[group][field] for field in FIELD_NAMES),
        )
        for group in GROUPS
    ]
    widths = tuple(max(len(header), *(len(str(row[idx])) for row in rows)) for idx, header in enumerate(headers))

    header_line = "  ".join(
        header.ljust(widths[i]) if i == 0 else header.rjust(widths[i]) for i, header in enumerate(headers)
    )
    print(header_line)

    for row in rows:
        row_line = "  ".join(
            str(val).ljust(widths[i]) if i == 0 else str(val).rjust(widths[i]) for i, val in enumerate(row)
        )
        print(row_line)

    print("\nPor comerciante:")
    if not merchant_counts:
        print("(sem registos)")
        return
    for merchant, count in merchant_counts.items():
        print(f"{merchant}: {count}")


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """Load the configured ground-truth CSV and print corpus statistics.

    Args:
        argv: Optional CLI arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom settings instance.
    """
    from src.golden_set import GoldenSetError, carregar_ground_truth
    from src.settings import get_settings

    settings = settings_override or get_settings()
    parser = argparse.ArgumentParser(description="Summarize receipt corpus statistics")
    parser.add_argument(
        "--csv",
        type=Path,
        default=settings.ground_truth_path,
        help="Path to the ground-truth CSV file",
    )
    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    try:
        records = carregar_ground_truth(args.csv)
    except GoldenSetError as exc:
        for err in exc.args[0]:
            print(err)
        return

    imprimir_resumo(records)


if __name__ == "__main__":
    main()
