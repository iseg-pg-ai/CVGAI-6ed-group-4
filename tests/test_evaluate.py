"""Offline tests for the end-to-end benchmark runner."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.evaluate import avaliar_corpus, avaliar_registo, main
from src.preprocessing import PreprocessOptions
from src.settings import load_settings
from src.types import BBox, Extraction, GroundTruthRecord, OcrDocument, OcrLine

OCR_TEXT = ["PINGO DOCE, S.A.", "NIF: 500 100 144", "Data: 26-10-27", "TOTAL 16,10 EUR"]
LLM_PAYLOAD = {
    "entidade": "PINGO DOCE, S.A.",
    "nif_comerciante": "500100144",
    "nif_cliente": "",
    "data": "2026-10-27",
    "total": "16.10",
    "presente": {
        "entidade": True,
        "nif_comerciante": True,
        "nif_cliente": False,
        "data": True,
        "total": True,
    },
    "linhas": {
        "entidade": 1,
        "nif_comerciante": 2,
        "nif_cliente": 0,
        "data": 3,
        "total": 4,
    },
}


def _document() -> OcrDocument:
    """Create the deterministic OCR document used throughout these tests."""
    lines = [
        OcrLine(index, text, 0.9, BBox(0, (index - 1) * 20, 200, index * 20))
        for index, text in enumerate(OCR_TEXT, start=1)
    ]
    return OcrDocument(lines=lines, engine="easyocr", elapsed_ms=0.1, image_size=(200, 80))


class FakeOcrEngine:
    """OCR double returning a fixed receipt document."""

    name: str = "easyocr"

    def __init__(self, **_: Any) -> None:
        self.read_calls = 0

    def read(self, image: np.ndarray) -> OcrDocument:
        self.read_calls += 1
        return _document()


class FakeLLMClient:
    """Minimal LLM client double returning a valid extraction payload."""

    provider = "llm_local"
    last_effective_provider = "llm_local"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        self.calls += 1
        return json.dumps(LLM_PAYLOAD)


def _record(filename: str = "receipt.jpg", group: str = "direito") -> GroundTruthRecord:
    """Build the ground truth matching the fake OCR document."""
    return GroundTruthRecord(
        filename=filename,
        grupo=group,  # type: ignore[arg-type]
        entidade="PINGO DOCE, S.A.",
        nif_comerciante="500100144",
        nif_cliente=None,
        data=date(2026, 10, 27),
        total=Decimal("16.10"),
    )


def _write_image(path: Path) -> None:
    """Write a small synthetic image file."""
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


@pytest.fixture(autouse=True)
def no_qr_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the QR stage deterministic while preserving its call in the pipeline."""
    monkeypatch.setattr("src.qr_reader.read", lambda image, raw_image=None: None)


def test_avaliar_registo_regex_pipeline_completo(tmp_path: Path) -> None:
    """The runner preprocesses, OCRs, extracts and scores all five regex fields."""
    image_path = tmp_path / "receipt.jpg"
    _write_image(image_path)
    engine = FakeOcrEngine()

    extraction, matches = avaliar_registo(
        _record(),
        tmp_path,
        engine,
        preprocess_opts=PreprocessOptions(
            rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False
        ),
    )

    assert isinstance(extraction, Extraction)
    assert extraction.method == "regex"
    assert engine.read_calls == 1
    assert len(matches) == 5
    assert all(match.acertou for match in matches)


def test_avaliar_registo_llm_com_cliente_injetado(tmp_path: Path) -> None:
    """The injected LLM client is passed to the LLM extractor without network access."""
    _write_image(tmp_path / "receipt.jpg")
    client = FakeLLMClient()

    extraction, matches = avaliar_registo(
        _record(),
        tmp_path,
        FakeOcrEngine(),
        method="llm_local",
        llm_client=client,
        preprocess_opts=PreprocessOptions(
            rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False
        ),
    )

    assert extraction.method == "llm_local"
    assert client.calls == 1
    assert len(matches) == 5
    assert all(match.acertou for match in matches)


def test_avaliar_registo_ficheiro_inexistente_levanta_erro(tmp_path: Path) -> None:
    """A missing receipt fails with the documented path-specific error."""
    with pytest.raises(FileNotFoundError, match="Receipt image not found"):
        avaliar_registo(_record("missing.jpg"), tmp_path, FakeOcrEngine())


def test_avaliar_registo_imagem_nao_descodificavel_levanta_erro(tmp_path: Path) -> None:
    """An existing but undecodable file raises the documented ValueError."""
    (tmp_path / "receipt.jpg").write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to decode receipt image"):
        avaliar_registo(_record(), tmp_path, FakeOcrEngine())


def test_avaliar_corpus_escreve_metrics_csv_e_md(tmp_path: Path) -> None:
    """Corpus evaluation scores both groups and writes the two report artifacts."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt-a.jpg")
    _write_image(receipts_dir / "receipt-b.jpg")
    records = [_record("receipt-a.jpg", "direito"), _record("receipt-b.jpg", "amarrotado")]
    output_dir = tmp_path / "results"

    results = avaliar_corpus(
        records,
        receipts_dir,
        FakeOcrEngine(),
        preprocess_opts=PreprocessOptions(
            rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False
        ),
        destino_resultados=output_dir,
    )

    assert len(results) == 10
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "metrics.md").is_file()
    md_text = (output_dir / "metrics.md").read_text(encoding="utf-8")
    assert "- **Motor OCR:** easyocr" in md_text
    assert "- **Método de extração:** regex" in md_text
    assert "- **Documentos avaliados:** 2" in md_text


def test_avaliar_corpus_reutiliza_um_cliente_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-regex provider is built once before iterating across receipt records."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt-a.jpg")
    _write_image(receipts_dir / "receipt-b.jpg")
    client = FakeLLMClient()
    builds: list[str] = []

    def build_client(provider: str, settings: Any = None) -> FakeLLMClient:
        builds.append(provider)
        return client

    monkeypatch.setattr("src.llm_client.build_client", build_client)

    results = avaliar_corpus(
        [_record("receipt-a.jpg"), _record("receipt-b.jpg", "amarrotado")],
        receipts_dir,
        FakeOcrEngine(),
        method="llm_local",
        preprocess_opts=PreprocessOptions(
            rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False
        ),
    )

    assert len(results) == 10
    assert builds == ["llm_local"]
    assert client.calls == 2


def _write_ground_truth(path: Path) -> None:
    """Write the one-row CSV consumed by the CLI test."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "grupo", "entidade", "nif_comerciante", "nif_cliente", "data", "total"])
        writer.writerow(["receipt.jpg", "direito", "PINGO DOCE, S.A.", "500100144", "", "2026-10-27", "16.10"])


def test_main_cli_executa_offline_com_doubles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI honors custom paths, uses the injected OCR double, and prints metrics."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt.jpg")
    csv_path = tmp_path / "truth.csv"
    _write_ground_truth(csv_path)
    output_dir = tmp_path / "results"
    settings = load_settings(repo_root=tmp_path)
    settings = replace(
        settings,
        ground_truth_path=csv_path,
        receipts_dir=receipts_dir,
        results_dir=output_dir,
    )
    monkeypatch.setattr("src.ocr_engine.EasyOCREngine", FakeOcrEngine)

    main(
        argv=[
            "--csv",
            str(csv_path),
            "--receipts-dir",
            str(receipts_dir),
            "--output-dir",
            str(output_dir),
            "--no-preprocess",
        ],
        settings_override=settings,
    )

    output = capsys.readouterr().out
    assert "Taxa de acerto global" in output
    assert "Taxa de falsos positivos" in output
    assert "Taxa de omissão" in output
    assert "Taxa de acerto (direito)" in output
    assert (output_dir / "metrics.md").is_file()


def test_main_cli_golden_set_invalido_imprime_erros(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the ground-truth CSV is invalid, main prints errors without crashing."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("invalid,header\n", encoding="utf-8")
    settings = load_settings(repo_root=tmp_path)

    main(
        argv=["--csv", str(bad_csv), "--receipts-dir", str(tmp_path)],
        settings_override=settings,
    )

    output = capsys.readouterr().out
    assert "Cabeçalho inválido" in output


def test_main_cli_engine_tesseract_e_metodo_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI selects Tesseract and LLM client doubles and runs successfully."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt.jpg")
    csv_path = tmp_path / "truth.csv"
    _write_ground_truth(csv_path)
    output_dir = tmp_path / "results"
    settings = load_settings(repo_root=tmp_path)
    settings = replace(
        settings,
        ground_truth_path=csv_path,
        receipts_dir=receipts_dir,
        results_dir=output_dir,
    )

    monkeypatch.setattr("src.ocr_tesseract.TesseractEngine", FakeOcrEngine)
    client = FakeLLMClient()
    monkeypatch.setattr("src.llm_client.build_client", lambda *args, **kwargs: client)

    main(
        argv=[
            "--csv",
            str(csv_path),
            "--receipts-dir",
            str(receipts_dir),
            "--output-dir",
            str(output_dir),
            "--engine",
            "tesseract",
            "--method",
            "llm_local",
            "--no-preprocess",
        ],
        settings_override=settings,
    )

    output = capsys.readouterr().out
    assert "Taxa de acerto global: 1.0000" in output
    assert client.calls == 1


def test_main_cli_erro_llm_tratado_sem_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A connection error from the LLM client is caught and printed cleanly."""
    from src.llm_client import LLMConnectionError

    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt.jpg")
    csv_path = tmp_path / "truth.csv"
    _write_ground_truth(csv_path)
    output_dir = tmp_path / "results"
    settings = load_settings(repo_root=tmp_path)
    settings = replace(
        settings,
        ground_truth_path=csv_path,
        receipts_dir=receipts_dir,
        results_dir=output_dir,
    )

    monkeypatch.setattr("src.ocr_engine.EasyOCREngine", FakeOcrEngine)

    class ErrorClient:
        def complete(self, *args: Any, **kwargs: Any) -> str:
            raise LLMConnectionError("Connection refused on 8080")

    monkeypatch.setattr("src.llm_client.build_client", lambda *args, **kwargs: ErrorClient())

    main(
        argv=[
            "--csv",
            str(csv_path),
            "--receipts-dir",
            str(receipts_dir),
            "--output-dir",
            str(output_dir),
            "--method",
            "llm_local",
            "--no-preprocess",
        ],
        settings_override=settings,
    )

    output = capsys.readouterr().out
    assert "Erro na extração com modelo de linguagem" in output
    assert "Connection refused on 8080" in output


def test_avaliar_corpus_invoca_progress_callback(tmp_path: Path) -> None:
    """Progress callback receives 1-based index, total count, and current record."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt-a.jpg")
    _write_image(receipts_dir / "receipt-b.jpg")
    records = [_record("receipt-a.jpg", "direito"), _record("receipt-b.jpg", "amarrotado")]

    calls: list[tuple[int, int, str]] = []

    def callback(idx: int, total: int, rec: GroundTruthRecord) -> None:
        calls.append((idx, total, rec.filename))

    avaliar_corpus(
        records,
        receipts_dir,
        FakeOcrEngine(),
        preprocess_opts=PreprocessOptions(
            rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False
        ),
        progress_callback=callback,
    )

    assert calls == [(1, 2, "receipt-a.jpg"), (2, 2, "receipt-b.jpg")]


def test_main_cli_com_flag_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI limit flag restricts evaluation to the specified number of records."""
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    _write_image(receipts_dir / "receipt-a.jpg")
    _write_image(receipts_dir / "receipt-b.jpg")

    csv_path = tmp_path / "truth.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "grupo", "entidade", "nif_comerciante", "nif_cliente", "data", "total"])
        writer.writerow(["receipt-a.jpg", "direito", "A", "500100144", "", "2026-10-27", "16.10"])
        writer.writerow(["receipt-b.jpg", "amarrotado", "B", "500100144", "", "2026-10-27", "16.10"])

    output_dir = tmp_path / "results"
    settings = load_settings(repo_root=tmp_path)
    settings = replace(
        settings,
        ground_truth_path=csv_path,
        receipts_dir=receipts_dir,
        results_dir=output_dir,
    )
    engine = FakeOcrEngine()
    monkeypatch.setattr("src.ocr_engine.EasyOCREngine", lambda *args, **kwargs: engine)

    main(
        argv=[
            "--csv",
            str(csv_path),
            "--receipts-dir",
            str(receipts_dir),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--no-preprocess",
        ],
        settings_override=settings,
    )

    output = capsys.readouterr().out
    assert "Limite ativo: a avaliar apenas os primeiros 1 registos." in output
    assert "[01/01] A avaliar receipt-a.jpg" in output
    assert engine.read_calls == 1
