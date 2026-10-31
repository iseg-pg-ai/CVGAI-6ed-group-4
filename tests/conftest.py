"""Shared pytest fixtures.

Every fixture here is offline, deterministic and leaves nothing behind: temporary artefacts
go to ``tmp_path`` and the repository is never written to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.types import Extraction, FieldValue

#: Header of ``data/ground_truth.csv``, in the mandatory order.
GROUND_TRUTH_HEADER = "filename,grupo,entidade,nif_comerciante,nif_cliente,data,total"

#: Minimal but valid set of rows, used by the golden-set tests.
VALID_ROWS = [
    "01_plano_pingo_doce.jpg,plano,Pingo Doce,500100144,123456789,2026-10-24,16.10",
    "02_amarrotado_pingo_doce_par.jpg,amarrotado,Pingo Doce,500100144,,2026-10-24,16.10",
]

# NIF used above must satisfy the modulo-11 check digit; 500100144 is the standard example.


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root.

    Returns:
        The directory that contains ``pyproject.toml``.
    """
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def receipts_tmp(tmp_path: Path) -> Path:
    """Create a temporary receipts directory holding empty image files.

    The files have valid JPEG names but no content: the golden-set validator only checks
    existence, never pixels, so the tests stay fast and need no image decoder.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        The created directory, containing ``01_plano_pingo_doce.jpg`` and
        ``02_amarrotado_pingo_doce_par.jpg``.
    """
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for row in VALID_ROWS:
        (receipts / row.split(",")[0]).touch()
    return receipts


@pytest.fixture
def ground_truth_valido(tmp_path: Path) -> Path:
    """Write a minimal but valid ``ground_truth.csv``.

    It is coherent with :func:`receipts_tmp` only when both are requested by the same test,
    which is the intended usage.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        Path of the written CSV.
    """
    path = tmp_path / "ground_truth.csv"
    path.write_text("\n".join([GROUND_TRUTH_HEADER, *VALID_ROWS]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def extracao_vazia() -> Extraction:
    """An :class:`Extraction` with all five fields marked ``AUSENTE``.

    Useful as the "system read nothing" baseline when testing scoring: every field the
    reference truth does contain must then be counted as an omission, and no field may be
    counted as a false positive.

    Returns:
        The empty extraction.
    """
    ausente = FieldValue(value=None, status="AUSENTE")
    return Extraction(
        entidade=ausente,
        nif_comerciante=ausente,
        nif_cliente=ausente,
        data=ausente,
        total=ausente,
        method="test",
        elapsed_ms=0.0,
    )


@pytest.fixture
def imagem_branca() -> np.ndarray:
    """A small synthetic BGR image, so tests never touch ``data/receipts``.

    Returns:
        A 240x120 uint8 array filled with white.
    """
    return np.full((240, 120, 3), 255, dtype=np.uint8)


@pytest.fixture
def extracao_exemplo() -> Extraction:
    """An extraction with plausible values, for scoring and formatting tests.

    Returns:
        A complete extraction where all five fields were found and carry a line id.
    """
    return Extraction(
        entidade=FieldValue(value="Pingo Doce", status="ENCONTRADO", line_id=1, confidence=0.9),
        nif_comerciante=FieldValue(value="500100144", status="ENCONTRADO", line_id=3),
        nif_cliente=FieldValue(value="123456789", status="ENCONTRADO", line_id=5),
        data=FieldValue(value="2026-10-24", status="ENCONTRADO", line_id=7),
        total=FieldValue(value="16.10", status="ENCONTRADO", line_id=42),
        method="test",
        elapsed_ms=12.5,
    )
