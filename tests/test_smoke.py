"""Green-start test: proves the environment, the pytest configuration and the import path.

This test exists so that a failure on day 0 is unambiguously an environment problem, and a
failure on day 8 is unambiguously a code problem. Keep it fast, offline and dependency-free
beyond the core package.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import src
from src.types import (
    CONSUMIDOR_FINAL_NIF,
    EARLIEST_PLAUSIBLE_DATE,
    EXTRACTION_METHODS,
    FIELD_NAMES,
    GROUPS,
    IMAGE_EXTENSIONS,
    LLM_PROVIDERS,
    OCR_ENGINES,
    Answer,
    BBox,
    Extraction,
    FieldValue,
    GroundTruthRecord,
    MatchResult,
    OcrDocument,
    OcrLine,
)


def test_package_is_importable() -> None:
    """The ``src`` package is reachable without installing the project."""
    assert src.__version__ == "0.1.0"


def test_bbox_geometry() -> None:
    """``BBox`` reports its own size and is immutable."""
    box = BBox(x1=10, y1=20, x2=40, y2=70)

    assert (box.width, box.height) == (30, 50), f"expected 30x50, got {box.width}x{box.height}"

    with pytest.raises(AttributeError):
        box.x1 = 0  # type: ignore[misc]

    with pytest.raises(ValueError):
        BBox(x1=50, y1=20, x2=40, y2=70)

    with pytest.raises(ValueError):
        BBox(x1=10, y1=80, x2=40, y2=70)


def test_ocr_document_renders_prompt_text() -> None:
    """Lines are numbered from one, zero-padded, in reading order."""
    document = OcrDocument(
        lines=[
            OcrLine(id=1, text="PINGO DOCE", confidence=0.98, bbox=BBox(0, 0, 10, 10)),
            OcrLine(id=2, text="TOTAL 16,10", confidence=0.91, bbox=BBox(0, 10, 10, 20)),
        ],
        engine="easyocr",
        elapsed_ms=1.0,
        image_size=(120, 240),
    )

    assert document.as_prompt_text() == "[01] PINGO DOCE\n[02] TOTAL 16,10"


def test_extraction_exposes_the_five_fields_in_order() -> None:
    """``Extraction.fields`` returns the assignment's five fields under their CSV names."""
    ausente = FieldValue(value=None, status="AUSENTE")
    extraction = Extraction(
        entidade=ausente,
        nif_comerciante=ausente,
        nif_cliente=ausente,
        data=ausente,
        total=ausente,
        method="regex",
        elapsed_ms=0.0,
    )

    assert list(extraction.fields) == ["entidade", "nif_comerciante", "nif_cliente", "data", "total"]


def test_ground_truth_record_keeps_absent_fields_as_none() -> None:
    """An absent field is ``None``, never a zero or a placeholder string."""
    record = GroundTruthRecord(
        filename="01_direito_pingo_doce.jpg",
        grupo="direito",
        entidade="Pingo Doce",
        nif_comerciante="500100144",
        nif_cliente=None,
        data=date(2026, 10, 24),
        total=Decimal("16.10"),
    )

    assert record.nif_cliente is None
    assert record.total == Decimal("16.10")


def test_match_result_and_answer_are_constructible() -> None:
    """The remaining contracts are constructible with their documented shapes."""
    match = MatchResult(
        filename="01_direito_pingo_doce.jpg",
        grupo="direito",
        campo="total",
        esperado="16.10",
        obtido="16.10",
        acertou=True,
        falso_positivo=False,
        omissao=False,
    )
    answer = Answer(
        question="Qual e o total?",
        answer="O total e 16,10 EUR.",
        status="ENCONTRADO",
        line_ids=[42],
        provider="llm_local",
    )

    assert match.acertou is True
    assert answer.status == "ENCONTRADO"


def test_numpy_is_available() -> None:
    """Numpy is a hard dependency of the contracts and must be importable."""
    assert np.zeros((2, 2), dtype=np.uint8).shape == (2, 2)


def test_repository_layout(repo_root: Path) -> None:
    """The scaffolding created by WP-00 is present."""
    assert (repo_root / "pyproject.toml").is_file()
    assert (repo_root / "src" / "types.py").is_file()
    assert (repo_root / "data" / "ground_truth.csv").is_file()


def test_shared_contract_constants() -> None:
    """Core domain constants are defined with correct types and values."""
    assert FIELD_NAMES == ("entidade", "nif_comerciante", "nif_cliente", "data", "total")
    assert GROUPS == ("direito", "amarrotado")
    assert OCR_ENGINES == ("easyocr", "tesseract")
    assert LLM_PROVIDERS == ("llm_local", "llm_gemini")
    assert EXTRACTION_METHODS == ("regex", "llm_local", "llm_gemini")
    assert CONSUMIDOR_FINAL_NIF == "999999990"
    assert date(2000, 1, 1) == EARLIEST_PLAUSIBLE_DATE
    assert IMAGE_EXTENSIONS == (".jpg", ".jpeg", ".png")
