"""Summarize the receipt corpus by image group, merchant, and field presence."""

from __future__ import annotations

from collections import Counter

from src.golden_set import carregar_ground_truth
from src.settings import get_settings
from src.types import GroundTruthRecord

_GROUPS = ("plano", "amarrotado")
_FIELDS = ("entidade", "nif_comerciante", "nif_cliente", "data", "total")
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
    group_counts: dict[str, dict[str, int]] = {group: {"n": 0, **dict.fromkeys(_FIELDS, 0)} for group in _GROUPS}
    merchant_counts: Counter[str] = Counter()
    field_counts = dict.fromkeys(_FIELDS, 0)

    for record in registos:
        group_counts[record.grupo]["n"] += 1
        merchant_counts[record.entidade or "(sem entidade)"] += 1
        for field in _FIELDS:
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

    field_order = _FIELDS
    headers = ("Grupo", "N", *(_FIELD_LABELS[field] for field in field_order))
    rows = [
        (
            group,
            group_counts[group]["n"],
            *(group_counts[group][field] for field in field_order),
        )
        for group in _GROUPS
    ]
    widths = tuple(max(len(header), *(len(str(row[index])) for row in rows)) for index, header in enumerate(headers))

    print(" ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rows:
        print(" ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))

    print("\nPor comerciante:")
    if not merchant_counts:
        print("(sem registos)")
        return
    for merchant, count in merchant_counts.items():
        print(f"{merchant}: {count}")


def main() -> None:
    """Load the configured ground-truth CSV and print corpus statistics."""
    settings = get_settings()
    imprimir_resumo(carregar_ground_truth(settings.ground_truth_path))


if __name__ == "__main__":
    main()
