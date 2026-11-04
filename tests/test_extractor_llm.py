"""Offline tests for the LLM extractor.

Every test injects a fake client that satisfies :class:`src.extraction.llm_client.LLMClient`; no request
leaves the process. The payloads are the shapes a model actually returns, including the two
failure modes that matter: a contradiction between ``presente`` and the value, and a line
number that does not exist in the document.

Ref: Docs/implementation_plan.md, section 7, WP-10.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.core.types import BBox, OcrDocument, OcrLine
from src.extraction.llm import FIELD_NAMES, SCHEMA, SYSTEM_PROMPT, LLMResponseError, extract, main
from src.extraction.llm_client import LLMClient, LLMConnectionError

#: A complete response, used as the base for the negative cases.
RECEIPT = {
    "entidade": "PINGO DOCE",
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


class FakeClient:
    """Minimal client double satisfying the ``LLMClient`` protocol."""

    def __init__(
        self,
        response: dict[str, Any] | str,
        *,
        provider: str = "llm_local",
        effective: str | None = None,
    ) -> None:
        """Store the canned response and record every call.

        Args:
            response: Payload dict, or raw text returned verbatim.
            provider: Advertised provider name.
            effective: Provider that answered, when it differs (failover).
        """
        self.response = response if isinstance(response, str) else json.dumps(response)
        self.provider = provider
        self.last_effective_provider = effective if effective is not None else provider
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        """Record the request and return the canned response."""
        self.calls.append({"system": system, "user": user, "schema": schema})
        return self.response


def _document(texts: list[str]) -> OcrDocument:
    """Build a synthetic OCR document with sequential line identifiers.

    Args:
        texts: Line texts in reading order.

    Returns:
        The document, with 20-pixel-tall boxes filled to the page.
    """
    lines = [
        OcrLine(
            id=index,
            text=text,
            confidence=0.9,
            bbox=BBox(0, (index - 1) * 20, 1000, (index - 1) * 20 + 18),
        )
        for index, text in enumerate(texts, start=1)
    ]
    return OcrDocument(lines=lines, engine="easyocr", elapsed_ms=1.0, image_size=(1000, 20 * max(1, len(texts))))


#: The receipt the canned payload refers to, with four lines so its line ids resolve.
RECEIPT_DOC = ["PINGO DOCE", "NIF: 500100144", "26-10-27", "TOTAL 16,10"]


def _variant(**changes: object) -> dict[str, Any]:
    """Return a deep copy of :data:`RECEIPT` with top-level keys replaced.

    Args:
        **changes: Keys to overwrite in the copy.

    Returns:
        An independent payload dict.
    """
    payload = copy.deepcopy(RECEIPT)
    payload.update(changes)
    return payload


def test_extracts_all_five_fields() -> None:
    """The happy path maps every field with its status and supporting line."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(RECEIPT))

    assert extraction.entidade.value == "PINGO DOCE"
    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_cliente.value is None
    assert extraction.data.value == "2026-10-27"
    assert extraction.total.value == "16.10"

    assert extraction.entidade.line_id == 1
    assert extraction.nif_comerciante.line_id == 2
    assert extraction.nif_cliente.line_id is None
    assert extraction.data.line_id == 3
    assert extraction.total.line_id == 4


def test_statuses_match_the_presence_flags() -> None:
    """Found fields are ENCONTRADO and a field the model marks absent is AUSENTE."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(RECEIPT))

    assert extraction.entidade.status == "ENCONTRADO"
    assert extraction.nif_cliente.status == "AUSENTE"


def test_method_records_the_provider_that_answered() -> None:
    """After a failover the extraction is labelled with the provider that actually replied."""
    client = FakeClient(RECEIPT, provider="llm_gemini", effective="llm_local")

    extraction = extract(_document(RECEIPT_DOC), client=client)

    assert extraction.method == "llm_local"


def test_raw_response_is_preserved() -> None:
    """The raw text is kept for the report appendix."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(RECEIPT))

    assert extraction.raw_response is not None
    assert "PINGO DOCE" in extraction.raw_response


def test_prompt_carries_the_numbered_lines_and_the_schema() -> None:
    """The user message is the indexed OCR text and the schema is passed separately."""
    client = FakeClient(RECEIPT)
    doc = _document(RECEIPT_DOC)

    extract(doc, client=client)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["user"] == doc.as_prompt_text()
    assert "[01]" in call["user"]
    assert call["schema"] == SCHEMA
    assert call["system"] == SYSTEM_PROMPT


def test_prompt_describes_the_json_shape_in_text() -> None:
    """The shape must be in the prompt, because llama-server does not inject the schema.

    This is the mitigation for the documented risk: without the shape in words, a small model
    returns valid JSON with invented key names.
    """
    for name in FIELD_NAMES:
        assert name in SYSTEM_PROMPT, f"{name} is missing from the system prompt"
    assert "presente" in SYSTEM_PROMPT
    assert "linhas" in SYSTEM_PROMPT
    assert "999999990" in SYSTEM_PROMPT, "the final-consumer code must be explained in the prompt"


def test_prompt_forbids_calculating_a_value() -> None:
    """The prompt must forbid arithmetic, because a live model performed it.

    Measured against the local Qwen2.5-Coder-3B on a receipt that prints only two article lines
    and no total: the model added 0.80 + 2.40, answered 3.20 and set ``presente.total`` to true.
    That is a false positive no post-processing can catch, since the number is arithmetically
    sound; only the prompt can prevent it.
    """
    lowered = SYSTEM_PROMPT.lower()
    assert "não calcules" in lowered
    assert "não somes" in lowered
    assert "não estimes" in lowered, "the model also estimates values it cannot see"
    assert "impresso" in lowered, "the total must be required to be printed, not derived"


def test_prompt_requires_a_line_number_for_every_present_field() -> None:
    """Grounding must be demanded in words, because the model otherwise answers 0.

    The same live run returned 0 for four of five fields on a long receipt, which would leave
    the interface with nothing to highlight. The prompt states the rule and the code re-derives
    the line as a backstop.
    """
    lowered = SYSTEM_PROMPT.lower()
    assert "maior que 0" in lowered
    assert "linhas" in lowered


def test_schema_is_flat_and_closed() -> None:
    """The schema uses no unions and closes every object, as llama.cpp and strict mode require."""
    assert SCHEMA["type"] == "object"
    assert SCHEMA["additionalProperties"] is False
    assert sorted(SCHEMA["required"]) == sorted([*FIELD_NAMES, "presente", "linhas"])

    for name in FIELD_NAMES:
        assert SCHEMA["properties"][name] == {"type": "string"}

    assert SCHEMA["properties"]["presente"]["type"] == "object"
    assert SCHEMA["properties"]["presente"]["additionalProperties"] is False
    assert SCHEMA["properties"]["linhas"]["type"] == "object"

    assert "anyOf" not in json.dumps(SCHEMA), "unions break json-schema-to-grammar"


def test_absent_field_is_not_a_value() -> None:
    """An empty string plus a false flag is AUSENTE, never a value."""
    payload = _variant(entidade="", nif_cliente="")
    payload["presente"]["entidade"] = False

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.value is None
    assert extraction.entidade.status == "AUSENTE"
    assert extraction.entidade.line_id is None


def test_empty_string_with_true_flag_is_absent() -> None:
    """A true flag with no value still cannot produce a value."""
    payload = _variant(entidade="")

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.status == "AUSENTE"


def test_contradiction_between_flag_and_value_resolves_to_absent() -> None:
    """When the model says absent but returns text, absence wins.

    This is the guard against a model that fills every field to look helpful; honouring the
    text would turn a document without a customer NIF into a false positive.
    """
    payload = copy.deepcopy(RECEIPT)
    payload["nif_cliente"] = "123456789"
    payload["presente"]["nif_cliente"] = False

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.nif_cliente.value is None
    assert extraction.nif_cliente.status == "AUSENTE"


def test_invalid_nif_is_marked_invalid_and_keeps_its_digits() -> None:
    """A nine-digit NIF with a broken check digit is INVALIDO, not silently accepted."""
    payload = _variant(nif_comerciante="518396895")

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.nif_comerciante.value == "518396895"
    assert extraction.nif_comerciante.status == "INVALIDO"


def test_nif_is_canonicalised_to_nine_digits() -> None:
    """Spaces in a printed NIF are removed before it reaches the comparison."""
    payload = _variant(nif_comerciante="500 100 144")

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.nif_comerciante.value == "500100144"
    assert extraction.nif_comerciante.status == "ENCONTRADO"


def test_consumer_final_sentinel_is_absent() -> None:
    """The AT final-consumer code is not a customer NIF, even though it validates."""
    payload = copy.deepcopy(RECEIPT)
    payload["nif_cliente"] = "999999990"
    payload["presente"]["nif_cliente"] = True

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.nif_cliente.value is None
    assert extraction.nif_cliente.status == "AUSENTE"


def test_nif_without_nine_digits_is_invalid() -> None:
    """A blob that is not a NIF at all is INVALIDO and keeps the text for the report."""
    payload = _variant(nif_comerciante="50010014")

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.nif_comerciante.value == "50010014"
    assert extraction.nif_comerciante.status == "INVALIDO"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-10-27", "2026-10-27"),
        ("27/10/2026", "2026-10-27"),
        ("26-10-27", "2026-10-27"),
        ("20261027", "2026-10-27"),
    ],
)
def test_date_is_reparsed_into_iso_form(raw: str, expected: str) -> None:
    """Whatever the model returns, the date is normalised by the pure parser."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(data=raw)))

    assert extraction.data.value == expected
    assert extraction.data.status == "ENCONTRADO"


def test_spelled_out_date_is_invalid() -> None:
    """A date written in words is INVALIDO under the current normaliser.

    The prompt asks for ``AAAA-MM-DD``, so a model that answers "27 de outubro de 2026" has
    broken the output contract and is reported as invalid rather than guessed at. Widening
    :func:`src.core.normalizers.parse_data` to month names is a WP-02 change, not a WP-10 one.
    """
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(data="27 de outubro de 2026")))

    assert extraction.data.status == "INVALIDO"
    assert extraction.data.value == "27 de outubro de 2026"


def test_impossible_date_is_invalid() -> None:
    """A date the calendar cannot hold is reported as INVALIDO, not as a value."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(data="31/02/2026")))

    assert extraction.data.status == "INVALIDO"
    assert extraction.data.value == "31/02/2026"


def test_future_date_is_invalid() -> None:
    """An issue date in the future is a misread and must not be reported as found."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(data="2099-12-31")))

    assert extraction.data.status == "INVALIDO"


def test_total_with_currency_and_comma_is_normalised() -> None:
    """The model may echo the printed amount; the stored value is canonical."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(total="16,10 EUR")))

    assert extraction.total.value == "16.10"
    assert extraction.total.status == "ENCONTRADO"


def test_zero_total_is_invalid() -> None:
    """The ground-truth contract needs a positive total, so zero is a misread."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(total="0.00")))

    assert extraction.total.status == "INVALIDO"


def test_unparseable_total_is_invalid() -> None:
    """Text where a number should be is INVALIDO and keeps its text."""
    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(_variant(total="dez euros")))

    assert extraction.total.status == "INVALIDO"
    assert extraction.total.value == "dez euros"


def test_fenced_json_with_prose_is_recovered() -> None:
    """Small models wrap the object; the parser must still find it."""
    raw = "Claro, aqui está:\n```json\n" + json.dumps(RECEIPT) + "\n```\nEspero ter ajudado."

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(raw))

    assert extraction.total.value == "16.10"


def test_response_without_json_raises() -> None:
    """Prose with no object is an error, not an empty extraction."""
    with pytest.raises(LLMResponseError, match="no JSON object"):
        extract(_document(RECEIPT_DOC), client=FakeClient("Desculpa, não consigo ler isto."))


def test_response_missing_a_key_raises() -> None:
    """A wrong-shaped object is rejected instead of degraded into five omissions."""
    payload = copy.deepcopy(RECEIPT)
    del payload["total"]

    with pytest.raises(LLMResponseError, match="missing keys: total"):
        extract(_document(RECEIPT_DOC), client=FakeClient(payload))


def test_response_with_non_object_flags_raises() -> None:
    """``presente`` must be an object; a list is a schema violation."""
    payload = _variant(presente=["entidade"])

    with pytest.raises(LLMResponseError, match="not an object"):
        extract(_document(RECEIPT_DOC), client=FakeClient(payload))


def test_unknown_keys_are_ignored() -> None:
    """An echoed schema or an extra field does not break the extraction."""
    payload = _variant()
    payload["confianca"] = 0.9

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.value == "PINGO DOCE"


def test_hallucinated_line_id_is_replaced_by_the_real_one() -> None:
    """A line number that does not exist is discarded, then the value is grounded again.

    The model is not trusted to count lines: on a real receipt it cited a non-existent line and
    on longer ones it cited nothing. The identifier is therefore re-derived from the text.
    """
    payload = copy.deepcopy(RECEIPT)
    payload["linhas"]["total"] = 99

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.total.value == "16.10"
    assert extraction.total.line_id == 4, "the total is printed on the last line"


def test_zero_line_id_is_replaced_by_the_real_one() -> None:
    """Zero means "no line cited"; the value is then located in the document.

    This is the measured failure mode of the local 3B model, which returned 0 for four of five
    fields on a long receipt while extracting the values correctly.
    """
    payload = copy.deepcopy(RECEIPT)
    payload["linhas"]["entidade"] = 0

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.value == "PINGO DOCE"
    assert extraction.entidade.line_id == 1


def test_grounding_is_skipped_when_the_model_cites_a_usable_line() -> None:
    """A valid identifier is trusted, so the model can still point at the right line."""
    payload = copy.deepcopy(RECEIPT)
    payload["linhas"]["entidade"] = 2

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.line_id == 2


def test_value_that_is_not_in_the_document_stays_ungrounded() -> None:
    """A value the text does not contain keeps no line instead of borrowing a wrong one."""
    payload = _variant(entidade="LOJA INVENTADA")
    payload["linhas"]["entidade"] = 0

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.entidade.value == "LOJA INVENTADA"
    assert extraction.entidade.line_id is None


@pytest.mark.parametrize(
    ("field", "value", "expected_line"),
    [
        ("nif_comerciante", "500 100 144", 2),
        ("data", "26-10-27", 3),
        ("total", "16,10 EUR", 4),
        ("total", "1 234,56", 5),
    ],
)
def test_grounding_handles_the_printed_forms(field: str, value: str, expected_line: int) -> None:
    """Each field is located using its own normaliser, not a raw string match.

    The document prints the values differently from the canonical form the module emits
    (spaced NIF, hyphenated date, comma decimal, grouped thousands), so a naive substring
    search would fail exactly where grounding matters.
    """
    texts = [
        "PINGO DOCE",
        "NIF: 500 100 144",
        "26-10-27",
        "TOTAL 16,10 EUR",
        "TOTAL 1 234,56",
    ]
    payload: dict[str, Any] = {
        "entidade": "PINGO DOCE",
        "nif_comerciante": value,
        "nif_cliente": "",
        "data": "26-10-27",
        "total": "1 234,56",
    }
    payload[field] = value
    payload["presente"] = dict.fromkeys(FIELD_NAMES, True)
    payload["presente"]["nif_cliente"] = False
    payload["linhas"] = dict.fromkeys(FIELD_NAMES, 0)

    extraction = extract(_document(texts), client=FakeClient(payload))

    assert extraction.fields[field].line_id == expected_line


def test_locate_line_total_prioritizes_total_line_over_article_line() -> None:
    """When an item price matches the total amount, grounding binds to the TOTAL line."""
    texts = [
        "CAFE CENTRAL",
        "1 CAFE 0,80",
        "1 PASTEL 0,80",
        "TOTAL 0,80",
    ]
    payload = {
        "entidade": "CAFE CENTRAL",
        "nif_comerciante": "",
        "nif_cliente": "",
        "data": "",
        "total": "0.80",
        "presente": {"entidade": True, "nif_comerciante": False, "nif_cliente": False, "data": False, "total": True},
        "linhas": {"entidade": 1, "nif_comerciante": 0, "nif_cliente": 0, "data": 0, "total": 0},
    }
    extraction = extract(_document(texts), client=FakeClient(payload))
    assert extraction.total.line_id == 4


@pytest.mark.parametrize("raw_line", [2, "2"])
def test_line_id_accepts_int_and_numeric_string(raw_line: object) -> None:
    """Providers disagree on the type of a line number; both spellings resolve."""
    payload = copy.deepcopy(RECEIPT)
    payload["linhas"]["total"] = raw_line

    extraction = extract(_document(RECEIPT_DOC), client=FakeClient(payload))

    assert extraction.total.line_id == 2


def test_empty_document_does_not_call_the_model() -> None:
    """A blank scan costs no request and yields five absent fields."""
    client = FakeClient(RECEIPT)
    doc = OcrDocument(lines=[], engine="easyocr", elapsed_ms=0.0, image_size=(10, 10))

    extraction = extract(doc, client=client)

    assert client.calls == []
    assert extraction.method == "llm_local"
    for name, field in extraction.fields.items():
        assert field.status == "AUSENTE", f"{name} was {field.status}"
    assert extraction.raw_response is None


def test_fake_client_satisfies_the_protocol() -> None:
    """The double used by these tests is a valid client, so the contract is exercised."""
    client: LLMClient = FakeClient(RECEIPT)

    assert client.provider == "llm_local"


def test_main_cli_prints_the_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The preview runs end to end with a fake OCR engine and a fake client."""
    img_path = tmp_path / "receipt.jpg"
    cv2.imwrite(str(img_path), np.full((120, 240, 3), 210, dtype=np.uint8))

    class FakeEngine:
        name = "easyocr"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return _document(RECEIPT_DOC)

    monkeypatch.setattr("src.ocr.easyocr.EasyOCREngine", FakeEngine)
    monkeypatch.setattr("src.extraction.llm_client.build_client", lambda *_a, **_k: FakeClient(RECEIPT))

    main(argv=["--file", str(img_path), "--engine", "easyocr", "--no-preprocess"])

    captured = capsys.readouterr().out
    assert "PINGO DOCE" in captured, captured
    assert "16.10" in captured, captured
    assert "llm_local" in captured, captured


def test_main_cli_reports_an_unreachable_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stopped llama-server produces a message, not a traceback."""
    img_path = tmp_path / "receipt.jpg"
    cv2.imwrite(str(img_path), np.full((120, 240, 3), 210, dtype=np.uint8))

    class FakeEngine:
        name = "easyocr"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return _document(RECEIPT_DOC)

    class DownClient(FakeClient):
        def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
            raise LLMConnectionError("Connection refused.")

    monkeypatch.setattr("src.ocr.easyocr.EasyOCREngine", FakeEngine)
    monkeypatch.setattr("src.extraction.llm_client.build_client", lambda *_a, **_k: DownClient(RECEIPT))

    main(argv=["--file", str(img_path), "--engine", "easyocr", "--no-preprocess"])

    assert "unavailable" in capsys.readouterr().out.lower()


def test_main_cli_reports_an_unusable_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A response that is not JSON produces a message, not a traceback."""
    img_path = tmp_path / "receipt.jpg"
    cv2.imwrite(str(img_path), np.full((120, 240, 3), 210, dtype=np.uint8))

    class FakeEngine:
        name = "easyocr"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def read(self, image: np.ndarray) -> OcrDocument:
            return _document(RECEIPT_DOC)

    monkeypatch.setattr("src.ocr.easyocr.EasyOCREngine", FakeEngine)
    monkeypatch.setattr("src.extraction.llm_client.build_client", lambda *_a, **_k: FakeClient("sem json"))

    main(argv=["--file", str(img_path), "--engine", "easyocr", "--no-preprocess"])

    assert "unusable" in capsys.readouterr().out.lower()


def test_reconcile_overrides_dropped_decimal_total() -> None:
    """When the model outputs a large integer total > 500 without decimals, valid regex decimal total wins."""
    texts = ["SUPERMERCADO", "NIF: 500100144", "26-10-27", "TOTAL 14,99 EUR"]
    payload = copy.deepcopy(RECEIPT)
    payload["total"] = "14096.00"

    extraction = extract(_document(texts), client=FakeClient(payload))
    assert extraction.total.value == "14.99"
    assert extraction.total.status == "ENCONTRADO"


def test_reconcile_drops_telecom_helpline_client_nif() -> None:
    """When the LLM outputs a toll-free 808 customer support number, reconcile resets it to AUSENTE."""
    texts = ["PINGO DOCE", "NIF: 500100144", "Linha Apoio Cliente 808204545", "TOTAL 5,86"]
    payload = copy.deepcopy(RECEIPT)
    payload["nif_cliente"] = "808204545"
    payload["presente"]["nif_cliente"] = True
    payload["linhas"]["nif_cliente"] = 3

    extraction = extract(_document(texts), client=FakeClient(payload))
    assert extraction.nif_cliente.status == "AUSENTE"
    assert extraction.nif_cliente.value is None


def test_main_cli_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing --file path is reported instead of raising."""
    main(argv=["--file", str(tmp_path / "absent.jpg")])

    assert "not found" in capsys.readouterr().out.lower()
