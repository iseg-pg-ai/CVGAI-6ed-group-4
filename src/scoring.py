"""Score extraction results against the receipt ground truth.

Ref: Docs/Planning/implementation_plan.md, section 4.7.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

from src.normalizers import valores_equivalentes
from src.types import FIELD_NAMES, GROUPS, Extraction, GroundTruthRecord, MatchResult

_RESULT_COLUMNS = (
    "filename",
    "grupo",
    "campo",
    "esperado",
    "obtido",
    "acertou",
    "falso_positivo",
    "omissao",
)
_METRIC_NAMES = ("taxa_acerto", "taxa_falso_positivo", "taxa_omissao")


class MetricRates(TypedDict):
    """The three core metric rates for an evaluation group."""

    taxa_acerto: float
    taxa_falso_positivo: float
    taxa_omissao: float


AggregatedMetrics = TypedDict(
    "AggregatedMetrics",
    {
        "por_campo": dict[str, MetricRates],
        "por_grupo": dict[str, MetricRates],
        "global": MetricRates,
    },
)


def _as_text(value: str | date | Decimal | None) -> str | None:
    """Convert typed ground-truth values to comparable string representations."""
    if value is None:
        return None
    return str(value)


def pontuar_registo(verdade: GroundTruthRecord, extracao: Extraction) -> list[MatchResult]:
    """Compare the five fields and return one MatchResult per field.

    Args:
        verdade: Validated reference values for one receipt.
        extracao: Extracted field values for the same receipt.

    Returns:
        Five comparisons in canonical field order. Missing predictions for fields that
        are also absent in the reference count as correct; inventing a value for an
        absent field is separately marked as a false positive.
    """
    expected_values = {
        "entidade": verdade.entidade,
        "nif_comerciante": verdade.nif_comerciante,
        "nif_cliente": verdade.nif_cliente,
        "data": verdade.data,
        "total": verdade.total,
    }

    results: list[MatchResult] = []
    for field in FIELD_NAMES:
        field_val = extracao.fields[field]
        expected = _as_text(expected_values[field])
        obtained = field_val.value
        is_equiv = valores_equivalentes(field, expected, obtained)
        acertou = False if expected is not None and field_val.status in ("INVALIDO", "AUSENTE") else is_equiv

        results.append(
            MatchResult(
                filename=verdade.filename,
                grupo=verdade.grupo,
                campo=field,
                esperado=expected,
                obtido=obtained,
                acertou=acertou,
                falso_positivo=expected is None and obtained is not None,
                omissao=expected is not None and obtained is None,
            )
        )
    return results


def _metricas(resultados: Iterable[MatchResult]) -> MetricRates:
    """Calculate accuracy, false-positive, and omission rates for match results."""
    matches = list(resultados)
    expected_absent = sum(result.esperado is None for result in matches)
    expected_present = sum(result.esperado is not None for result in matches)
    return {
        "taxa_acerto": sum(result.acertou for result in matches) / len(matches) if matches else 0.0,
        "taxa_falso_positivo": (
            sum(result.falso_positivo for result in matches) / expected_absent if expected_absent else 0.0
        ),
        "taxa_omissao": sum(result.omissao for result in matches) / expected_present if expected_present else 0.0,
    }


def agregar(resultados: list[MatchResult]) -> AggregatedMetrics:
    """Aggregate accuracy, false-positive, and omission rates by field and group.

    ``taxa_acerto`` counts a missing value as correct when the reference is also
    absent and the extraction is ``AUSENTE``. False-positive rates use absent
    reference fields as their denominator, and omission rates use present reference
    fields. A rate is ``0.0`` when its denominator is zero.

    Args:
        resultados: Per-field comparisons, usually combined across receipts.

    Returns:
        The requested structure with ``por_campo``, ``por_grupo``, and ``global``.
        ``por_campo`` and ``por_grupo`` map each field/group to the three rates;
        ``global`` contains the rates over all results.
    """
    fields = (*FIELD_NAMES, *sorted({result.campo for result in resultados} - set(FIELD_NAMES)))
    groups = (*GROUPS, *sorted({result.grupo for result in resultados} - set(GROUPS)))
    return {
        "por_campo": {field: _metricas(result for result in resultados if result.campo == field) for field in fields},
        "por_grupo": {group: _metricas(result for result in resultados if result.grupo == group) for group in groups},
        "global": _metricas(resultados),
    }


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """Render a table as Markdown-formatted lines."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def escrever_tabelas(
    resultados: list[MatchResult],
    destino: Path,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write results/metrics.csv and results/metrics.md (table for the report).

    ``destino`` is the output directory; this function creates
    ``metrics.csv`` and ``metrics.md`` inside it.

    Args:
        resultados: Per-field comparisons to export.
        destino: Directory in which to write the two report files.
        metadata: Optional dictionary of run metadata to include in the Markdown header.
    """
    destino.mkdir(parents=True, exist_ok=True)
    csv_path = destino / "metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        for result in resultados:
            writer.writerow(
                {
                    "filename": result.filename,
                    "grupo": result.grupo,
                    "campo": result.campo,
                    "esperado": result.esperado if result.esperado is not None else "",
                    "obtido": result.obtido if result.obtido is not None else "",
                    "acertou": str(result.acertou).lower(),
                    "falso_positivo": str(result.falso_positivo).lower(),
                    "omissao": str(result.omissao).lower(),
                }
            )

    aggregated = agregar(resultados)
    markdown_lines = ["# Métricas do benchmark", ""]
    if metadata:
        for key, value in metadata.items():
            markdown_lines.append(f"- **{key}:** {value}")
        markdown_lines.append("")
    markdown_lines.extend(["## Global", ""])
    global_metrics = aggregated["global"]
    markdown_lines.extend(
        _format_table(
            ("Métrica", "Valor"),
            [(name, f"{global_metrics[name]:.4f}") for name in _METRIC_NAMES],
        )
    )
    for section, title in (("por_campo", "Por campo"), ("por_grupo", "Por grupo")):
        markdown_lines.extend(["", f"## {title}", ""])
        rows = [
            (key, *(f"{metrics[name]:.4f}" for name in _METRIC_NAMES)) for key, metrics in aggregated[section].items()
        ]
        markdown_lines.extend(_format_table(("Dimensão", *_METRIC_NAMES), rows))
    (destino / "metrics.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
