"""Offline tests for the regex baseline extractor.

Every case builds a synthetic :class:`OcrDocument`; no image is decoded and nothing touches
``data/receipts``. Several tests pin behaviours that are easy to regress and expensive to
notice in a benchmark: the consumer-final sentinel, the duplicate-NIF rule, and the fact that
:func:`src.normalizers.parse_moeda` would read a quantity as the total.

Ref: Docs/Planning/implementation_plan.md, section 5 (WP-09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.extractor_regex import CONSUMIDOR_FINAL_NIF, extract, main
from src.types import BBox, OcrDocument, OcrLine

#: Vertical step between synthetic lines.
SPACING = 20


def _document(texts: list[str], *, width: int = 1000, height: int | None = None) -> OcrDocument:
    """Build a synthetic OCR document from a list of line texts.

    Lines are numbered from 1 and stacked top to bottom, which is the reading order the real
    engines produce. The canvas is filled by default so the vertical positions are realistic
    and the header band used by the NIF fallback only covers the top of the page.

    Args:
        texts: Line texts in reading order.
        width: Image width in pixels.
        height: Image height in pixels. Defaults to :data:`SPACING` times the line count.

    Returns:
        An :class:`OcrDocument` with sequential identifiers and non-overlapping boxes.
    """
    canvas_height = height if height is not None else SPACING * max(1, len(texts))
    lines = [
        OcrLine(
            id=index, text=text, confidence=0.9, bbox=BBox(0, (index - 1) * SPACING, width, (index - 1) * SPACING + 18)
        )
        for index, text in enumerate(texts, start=1)
    ]
    return OcrDocument(lines=lines, engine="easyocr", elapsed_ms=1.0, image_size=(width, canvas_height))


#: A realistic flat supermarket receipt, used as the happy path.
SUPERMARKET = [
    "PINGO DOCE, S.A.",
    "RUA 25 DE ABRIL 12",
    "NIF: 500 100 144",
    "Contribuinte: 500100144",
    "FATURA SIMPLIFICADA",
    "Data: 26-10-27 13:45:42",
    "Arroz 1,29",
    "SUBTOTAL 13,08",
    "IVA 23% 3,02",
    "TOTAL IVA 3,02",
    "TOTAL 16,10 EUR",
]


def test_extracts_all_five_fields_from_a_flat_receipt() -> None:
    """The happy path finds every field with the expected canonical value."""
    extraction = extract(_document(SUPERMARKET))

    assert extraction.entidade.value == "PINGO DOCE, S.A."
    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_cliente.value is None
    assert extraction.data.value == "2026-10-27"
    assert extraction.total.value == "16.10"

    assert extraction.entidade.status == "ENCONTRADO"
    assert extraction.nif_comerciante.status == "ENCONTRADO"
    assert extraction.nif_cliente.status == "AUSENTE", "this receipt carries no customer NIF"
    assert extraction.data.status == "ENCONTRADO"
    assert extraction.total.status == "ENCONTRADO"


def test_values_use_the_canonical_forms_scoring_expects() -> None:
    """Total and date are emitted in the same shape as the ground-truth CSV columns."""
    extraction = extract(_document(SUPERMARKET))

    assert extraction.total.value == "16.10", "totals must use a dot and two decimals"
    assert extraction.data.value == "2026-10-27", "dates must use ISO 8601"


def test_line_ids_point_at_the_supporting_line() -> None:
    """Every found field cites the line it was read from."""
    extraction = extract(_document(SUPERMARKET))

    assert extraction.entidade.line_id == 1
    assert extraction.nif_comerciante.line_id == 3
    assert extraction.data.line_id == 6
    assert extraction.total.line_id == 11


def test_extraction_carries_method_and_timing() -> None:
    """The extraction is tagged as the regex baseline and reports a non-negative duration."""
    extraction = extract(_document(SUPERMARKET))

    assert extraction.method == "regex"
    assert extraction.elapsed_ms >= 0.0
    assert extraction.raw_response is None


def test_entity_skips_document_header_lines() -> None:
    """A 'FATURA SIMPLIFICADA' banner is not the merchant name."""
    extraction = extract(_document(["FATURA SIMPLIFICADA", "TALHO DO ZE", "NIF 500100144", "TOTAL 7,30"]))

    assert extraction.entidade.value == "TALHO DO ZE"
    assert extraction.entidade.line_id == 2


def test_entity_ignores_digit_heavy_lines() -> None:
    """A line made of codes and numbers is not a trading name."""
    extraction = extract(_document(["123456789 12", "LOJA DO ZE", "TOTAL 3,00"]))

    assert extraction.entidade.value == "LOJA DO ZE"


def test_entity_absent_when_no_plausible_name() -> None:
    """Without a name-like line the field is AUSENTE, not a guess."""
    extraction = extract(_document(["NIF: 500100144", "TOTAL 3,00"]))

    assert extraction.entidade.value is None
    assert extraction.entidade.status == "AUSENTE"


def test_merchant_nif_from_labelled_line() -> None:
    """An issuer label is the strongest signal for the merchant NIF."""
    extraction = extract(_document(["LOJA DO ZE", "RUA DAS FLORES 4", "NIF: 500 100 144", "TOTAL 1,00"]))

    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_comerciante.status == "ENCONTRADO"


def test_merchant_nif_falls_back_to_the_header_band() -> None:
    """An unlabelled NIF in the header is accepted as the issuer number."""
    extraction = extract(_document(["SUPERMERCADO CENTRAL", "500100144", "123456789", "TOTAL 9,00"]))

    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_cliente.value is None, "an unlabelled second NIF is not promoted to customer"


def test_merchant_nif_ignores_bare_numbers_below_the_header() -> None:
    """An unlabelled number at the foot of the page is not the issuer NIF.

    The band covers the top 40 % of the image; a bare number printed under the total belongs
    to a payment line or an order reference, not to the merchant block.
    """
    extraction = extract(_document(["LOJA DO ZE", "RUA DAS FLORES 4", "TOTAL 9,00", "500100144"]))

    assert extraction.nif_comerciante.value is None
    assert extraction.nif_comerciante.status == "AUSENTE"


def test_client_nif_requires_an_explicit_label() -> None:
    """The customer NIF is only read from a line that names the customer."""
    extraction = extract(_document(["LOJA DO ZE", "NIF: 500100144", "CLIENTE: 123456789", "TOTAL 2,00"]))

    assert extraction.nif_cliente.value == "123456789"
    assert extraction.nif_cliente.status == "ENCONTRADO"


def test_client_nif_absent_without_a_label() -> None:
    """Silence is reported as AUSENTE, never invented."""
    extraction = extract(_document(["LOJA DO ZE", "NIF: 500100144", "TOTAL 2,00"]))

    assert extraction.nif_cliente.value is None
    assert extraction.nif_cliente.status == "AUSENTE"


def test_consumer_final_sentinel_is_absent_not_found() -> None:
    """The AT final-consumer code must never surface as an extracted NIF.

    ``999999990`` satisfies the modulo-11 NIF check digit, so this guards the explicit
    sentinel rejection rather than the validator.
    """
    extraction = extract(_document(["CAFE CENTRAL", "999999990", "CLIENTE: 999999990", "TOTAL 2,50"]))

    assert extraction.nif_comerciante.value is None
    assert extraction.nif_cliente.value is None
    assert extraction.nif_comerciante.status == "AUSENTE"
    assert extraction.nif_cliente.status == "AUSENTE"


def test_sentinel_constant_matches_the_at_code() -> None:
    """The local constant documented in the module is the AT final-consumer code."""
    assert CONSUMIDOR_FINAL_NIF == "999999990"


def test_client_nif_is_never_the_merchant_nif() -> None:
    """When both lines carry the same number, the duplicate is dropped from the customer."""
    extraction = extract(_document(["PINGO DOCE", "NIF: 500100144", "CLIENTE: 500100144", "TOTAL 4,00"]))

    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_cliente.value is None
    assert extraction.nif_cliente.status == "AUSENTE"


def test_both_nifs_on_a_single_line() -> None:
    """A line mixing the two labels yields the first number to the merchant and the last to the customer."""
    extraction = extract(_document(["LOJA DO ZE", "NIF: 500100144 NIF CLIENTE: 123456789", "TOTAL 1,00"]))

    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_cliente.value == "123456789"


def test_invalid_nif_keeps_its_digits_and_is_marked_invalid() -> None:
    """A nine-digit number with a broken check digit is INVALIDO, not AUSENTE.

    Keeping the digits is what lets scoring count the field as wrong instead of as an
    omission.
    """
    extraction = extract(_document(["LOJA DO ZE", "NIF: 518396895", "TOTAL 5,00"]))

    assert extraction.nif_comerciante.value == "518396895"
    assert extraction.nif_comerciante.status == "INVALIDO"


def test_total_ignores_subtotal_and_tax_breakdown() -> None:
    """Only the amount due is taken from a receipt that prints three 'total' variants."""
    extraction = extract(_document(SUPERMARKET))

    assert extraction.total.value == "16.10"
    assert extraction.total.line_id == 11


def test_total_takes_the_last_matching_line() -> None:
    """When 'TOTAL' appears twice, the final occurrence wins."""
    extraction = extract(_document(["LOJA DO ZE", "TOTAL 13,08", "TOTAL 16,10"]))

    assert extraction.total.value == "16.10"
    assert extraction.total.line_id == 3


def test_total_qualifier_and_currency_prefix_variations() -> None:
    """Supports TOTAL EUR 16,10, TOTAL A PAGAR, and A PAGAR formats."""
    assert extract(_document(["LOJA", "TOTAL EUR 16,10"])).total.value == "16.10"
    assert extract(_document(["LOJA", "TOTAL A PAGAR 16,10"])).total.value == "16.10"
    assert extract(_document(["LOJA", "A PAGAR: 16,10"])).total.value == "16.10"
    assert extract(_document(["LOJA", "TOTAL GERAL 16,10 €"])).total.value == "16.10"


def test_greeting_line_ignored_for_entity() -> None:
    """Banner lines starting with BEM-VINDO or 2ª VIA are not picked as entity."""
    extraction = extract(_document(["BEM-VINDO", "PASTELARIA LISBOA", "NIF 501964843", "TOTAL 2,00"]))
    assert extraction.entidade.value == "PASTELARIA LISBOA"


def test_total_survives_a_thousands_separator() -> None:
    """Grouped thousands are read as one amount, not as two numbers."""
    extraction = extract(_document(["MERCEARIA", "NIF 500100144", "TOTAL 1 234,56"]))

    assert extraction.total.value == "1234.56"


def test_total_rejects_a_line_with_noise_between_label_and_amount() -> None:
    """A noisy total line is skipped instead of being misread.

    ``parse_moeda`` returns the first number it finds, so handing it the whole line would
    report the quantity as the total. This extractor requires the amount to close the line,
    which costs a real total here but never invents one; for a baseline scored on false
    positives that is the right trade, and this test pins the decision.
    """
    extraction = extract(_document(["MERCEARIA", "TOTAL 2 UN 1 234,56"]))

    assert extraction.total.value is None
    assert extraction.total.status == "AUSENTE"


def test_total_absent_when_no_labelled_total() -> None:
    """A page of amounts without a total label produces AUSENTE, not the largest number."""
    extraction = extract(_document(["LOJA DO ZE", "Arroz 1,29", "IVA 0,24"]))

    assert extraction.total.value is None
    assert extraction.total.status == "AUSENTE"


def test_total_zero_is_invalid() -> None:
    """The ground-truth contract requires a positive total, so zero is a misread."""
    extraction = extract(_document(["LOJA DO ZE", "TOTAL 0,00"]))

    assert extraction.total.value == "0.00"
    assert extraction.total.status == "INVALIDO"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Data: 2026-10-27", "2026-10-27"),
        ("26-10-27", "2026-10-27"),
        ("26-10-27 13:45:42", "2026-10-27"),
        ("20261027", "2026-10-27"),
        ("Data: 27/10/2026", "2026-10-27"),
    ],
)
def test_date_formats(text: str, expected: str) -> None:
    """Every shape the corpus prints is normalised to ISO 8601."""
    extraction = extract(_document(["LOJA DO ZE", text, "TOTAL 1,00"]))

    assert extraction.data.value == expected


def test_date_in_the_future_is_skipped() -> None:
    """An impossible issue date is ignored and the next plausible one is used.

    Hyphenated dates are read as YY-MM-DD, so a misread can land in the next decade; the
    upper bound is what keeps that out of the result.
    """
    extraction = extract(_document(["LOJA DO ZE", "2099-12-31", "2026-10-24", "TOTAL 9,99"]))

    assert extraction.data.value == "2026-10-24"
    assert extraction.data.line_id == 3


def test_date_absent_without_a_plausible_candidate() -> None:
    """A document holding only impossible dates reports AUSENTE."""
    extraction = extract(_document(["LOJA DO ZE", "2099-12-31", "TOTAL 9,99"]))

    assert extraction.data.value is None
    assert extraction.data.status == "AUSENTE"


def test_empty_document_returns_all_fields_absent() -> None:
    """An engine that read nothing produces a well-formed empty extraction."""
    extraction = extract(OcrDocument(lines=[], engine="easyocr", elapsed_ms=0.0, image_size=(10, 10)))

    for name, field in extraction.fields.items():
        assert field.status == "AUSENTE", f"{name} was {field.status}, expected AUSENTE"
        assert field.value is None


def test_qr_payload_is_never_used_to_fill_a_field() -> None:
    """The baseline is measured on OCR text alone, even when a perfect QR is supplied."""
    qr = {"total": "16.10", "data": "2026-10-27", "nif_comerciante": "500100144"}
    extraction = extract(_document(["LOJA DO ZE", "TOTAL 3,00"]), qr=qr)

    assert extraction.total.value == "3.00", "the OCR text must win over the QR payload"
    assert extraction.data.value is None, "the QR date must not populate the date field"
    assert extraction.nif_comerciante.status == "AUSENTE"


def test_main_cli_prints_the_extracted_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The preview runs end to end with a fake OCR engine and prints one row per field."""
    img_path = tmp_path / "receipt.jpg"
    cv2.imwrite(str(img_path), np.full((120, 240, 3), 210, dtype=np.uint8))

    class FakeEngine:
        name = "easyocr"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return _document(["PINGO DOCE", "NIF: 500100144", "TOTAL 16,10"])

    monkeypatch.setattr("src.ocr_engine.EasyOCREngine", FakeEngine)

    main(argv=["--file", str(img_path), "--engine", "easyocr", "--no-preprocess"])

    captured = capsys.readouterr().out
    assert "PINGO DOCE" in captured, captured
    assert "500100144" in captured, captured
    assert "16.10" in captured, captured
    assert "ENCONTRADO" in captured, captured
    assert "3 lines" in captured, captured


def test_main_cli_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing --file path is reported instead of raising."""
    main(argv=["--file", str(tmp_path / "absent.jpg")])

    assert "not found" in capsys.readouterr().out.lower()


def test_main_cli_missing_receipts_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An absent receipts directory is reported instead of raising."""
    from src.settings import load_settings

    settings = load_settings(repo_root=tmp_path)

    main(argv=[], settings_override=settings)

    assert "not found" in capsys.readouterr().out.lower()
