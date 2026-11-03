"""Unit tests for the evidence-grounded Q&A assistant module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.types import BBox, OcrDocument, OcrLine
from src.extraction.qa_assistant import QAResponseError, ask, main


class MockLLMClient:
    """Mock LLM client returning predetermined text for testing."""

    def __init__(self, response_text: str, provider: str = "llm_local") -> None:
        self.provider = provider
        self.last_effective_provider = provider
        self.response_text = response_text

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        return self.response_text


@pytest.fixture
def sample_ocr_doc() -> OcrDocument:
    """Create a sample OcrDocument for Q&A testing."""
    lines = [
        OcrLine(id=1, text="PINGO DOCE - SUPERMERCADOS", confidence=0.98, bbox=BBox(10, 10, 100, 30)),
        OcrLine(id=2, text="NIF: 500100144", confidence=0.95, bbox=BBox(10, 40, 90, 60)),
        OcrLine(id=3, text="TOTAL A PAGAR: 16,10 EUR", confidence=0.99, bbox=BBox(10, 100, 110, 130)),
    ]
    return OcrDocument(lines=lines, engine="easyocr", elapsed_ms=2.0, image_size=(150, 150))


def test_ask_found_with_valid_lines(sample_ocr_doc: OcrDocument) -> None:
    """ask returns ENCONTRADO and valid supporting line IDs when model answers successfully."""
    payload = {
        "answer": "O total a pagar é 16,10 EUR.",
        "status": "ENCONTRADO",
        "linhas": [3],
    }
    client = MockLLMClient(json.dumps(payload), provider="llm_local")

    answer = ask("Qual é o total?", sample_ocr_doc, client=client)

    assert answer.status == "ENCONTRADO"
    assert answer.line_ids == [3]
    assert "16,10" in answer.answer
    assert answer.provider == "llm_local"
    assert answer.question == "Qual é o total?"


def test_ask_empty_document() -> None:
    """ask handles empty OCR document gracefully without calling LLM."""
    doc = OcrDocument(lines=[], engine="easyocr", elapsed_ms=0.0, image_size=(100, 100))
    client = MockLLMClient("should not be called")

    answer = ask("Qual é o total?", doc, client=client)

    assert answer.status == "AUSENTE NO DOCUMENTO"
    assert answer.line_ids == []
    assert "nao contem" in answer.answer.lower() or "não contem" in answer.answer.lower()


def test_ask_absent_document(sample_ocr_doc: OcrDocument) -> None:
    """ask returns AUSENTE NO DOCUMENTO when the information is missing."""
    payload = {
        "answer": "O documento não contém o email da loja.",
        "status": "AUSENTE NO DOCUMENTO",
        "linhas": [],
    }
    client = MockLLMClient(json.dumps(payload))

    answer = ask("Qual é o email?", sample_ocr_doc, client=client)

    assert answer.status == "AUSENTE NO DOCUMENTO"
    assert answer.line_ids == []


def test_ask_reanchor_lines(sample_ocr_doc: OcrDocument) -> None:
    """ask re-anchors supporting lines if model returns ENCONTRADO but no valid lines."""
    payload = {
        "answer": "O NIF da empresa é 500100144.",
        "status": "ENCONTRADO",
        "linhas": [],  # Model failed to cite line
    }
    client = MockLLMClient(json.dumps(payload))

    answer = ask("Qual é o NIF?", sample_ocr_doc, client=client)

    # Re-anchored to line 2 because line 2 contains NIF or digits matching query/answer
    assert answer.status == "ENCONTRADO"
    assert len(answer.line_ids) > 0


def test_malformed_json_raises_qa_response_error(sample_ocr_doc: OcrDocument) -> None:
    """ask raises QAResponseError when response is not valid JSON."""
    client = MockLLMClient("I am not JSON at all")

    with pytest.raises(QAResponseError):
        ask("Qual é o total?", sample_ocr_doc, client=client)


def test_missing_keys_raises_qa_response_error(sample_ocr_doc: OcrDocument) -> None:
    """ask raises QAResponseError when required keys are missing from response."""
    payload = {"answer": "Some answer"}  # missing status and linhas
    client = MockLLMClient(json.dumps(payload))

    with pytest.raises(QAResponseError):
        ask("Qual é o total?", sample_ocr_doc, client=client)


def test_main_cli_preview(
    receipts_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI main runs cleanly with mocked OCR and LLM client."""
    from src.core.settings import load_settings
    from src.core.types import BBox, OcrDocument, OcrLine

    receipt_img = receipts_tmp / "test_receipt.jpg"
    import cv2
    import numpy as np

    cv2.imwrite(str(receipt_img), np.full((100, 100, 3), 255, dtype=np.uint8))

    override = load_settings(repo_root=receipts_tmp)

    fake_doc = OcrDocument(
        lines=[OcrLine(id=1, text="TOTAL 16.10", confidence=0.9, bbox=BBox(0, 0, 10, 10))],
        engine="easyocr",
        elapsed_ms=1.0,
        image_size=(100, 100),
    )
    monkeypatch.setattr("src.ocr.easyocr.EasyOCREngine.read", lambda self, img: fake_doc)
    monkeypatch.setattr(
        "src.extraction.llm_client.OpenAICompatibleClient.complete",
        lambda self, sys, user, schema=None, **kwargs: (
            '{"answer": "O total é 16.10", "status": "ENCONTRADO", "linhas": [1]}'
        ),
    )

    main(argv=["--file", str(receipt_img), "--question", "Total?"], settings_override=override)
    captured = capsys.readouterr().out
    assert "--- Q&A Result" in captured
