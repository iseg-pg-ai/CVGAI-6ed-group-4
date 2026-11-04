"""Extraction of the five required fields with a language model.

This is approach B of ``PROJECT_PLAN.md`` (section 4, level 2): the OCR supplies text and
coordinates, the model supplies semantic understanding, and the result comes back as strict
JSON. Works with every backend of :mod:`src.extraction.llm_client`, so the local and cloud runs differ
only in the client instance, never in the prompt or the post-processing.

Ref: Docs/implementation_plan.md, section 7, WP-10.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, cast

from src.core.cli_utils import format_extraction_field
from src.core.types import (
    FIELD_NAMES,
    LLM_PROVIDERS,
    OCR_ENGINES,
    Extraction,
    FieldValue,
    MethodName,
    OcrDocument,
)
from src.extraction.llm_grounding import (
    _as_text,
    _build_field,
    _field_from_payload,
    _line_id,
    _locate_line,
    _require_shape,
    as_text,
    build_field,
    field_from_payload,
    line_id_from_raw,
    locate_line,
    require_shape,
)
from src.extraction.llm_prompts import (
    SCHEMA,
    SYSTEM_PROMPT,
    LLMResponseError,
    _parse_json_object,
    parse_response_payload,
)

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.extraction.llm_client import LLMClient


logger = logging.getLogger(__name__)

__all__ = [
    "FIELD_NAMES",
    "LLMResponseError",
    "SCHEMA",
    "SYSTEM_PROMPT",
    "_absent_extraction",
    "_as_text",
    "_build_field",
    "_field_from_payload",
    "_line_id",
    "_locate_line",
    "_parse_json_object",
    "_reconcile_nifs",
    "_require_shape",
    "as_text",
    "build_field",
    "extract",
    "field_from_payload",
    "line_id_from_raw",
    "locate_line",
    "main",
    "parse_response_payload",
    "require_shape",
]


def _reconcile_nifs(
    merchant: FieldValue,
    client: FieldValue,
    doc: OcrDocument,
    payload: dict[str, Any],
) -> tuple[FieldValue, FieldValue]:
    """Apply OCR structural consistency rules to disambiguate merchant and customer NIFs."""
    from src.extraction.regex import _extract_nifs

    explicit_client_contradiction = (
        payload.get("presente", {}).get("nif_cliente") is False and _as_text(payload.get("nif_cliente", "")) != ""
    )

    regex_merchant, regex_client = _extract_nifs(doc.lines, doc.image_size)

    def _assign_client_and_fallback_merchant(new_client: FieldValue) -> tuple[FieldValue, FieldValue]:
        if regex_merchant.status == "ENCONTRADO" and regex_merchant.value != new_client.value:
            return regex_merchant, new_client
        return FieldValue(value=None, status="AUSENTE"), new_client

    if regex_client.value is not None and not explicit_client_contradiction:
        if merchant.value == regex_client.value or bool(merchant.value and merchant.value in regex_client.value):
            merchant, client = _assign_client_and_fallback_merchant(regex_client)
        elif client.status == "AUSENTE" and merchant.value != regex_client.value:
            client = regex_client
    if (
        merchant.status == "ENCONTRADO"
        and merchant.value is not None
        and merchant.value[0] in "123"
        and client.status == "AUSENTE"
        and not explicit_client_contradiction
    ):
        merchant, client = _assign_client_and_fallback_merchant(merchant)

    if (
        merchant.status == "INVALIDO"
        and client.status == "AUSENTE"
        and regex_client.status == "ENCONTRADO"
        and regex_client.value is not None
        and not explicit_client_contradiction
        and bool(merchant.value and merchant.value in regex_client.value)
    ):
        merchant, client = _assign_client_and_fallback_merchant(regex_client)

    if (
        client.status == "AUSENTE"
        and not explicit_client_contradiction
        and regex_client.status == "ENCONTRADO"
        and regex_client.value is not None
        and merchant.value != regex_client.value
    ):
        client = regex_client

    if (
        (merchant.status == "AUSENTE" or (merchant.status == "INVALIDO" and merchant.value and "-" in merchant.value))
        and regex_merchant.status == "ENCONTRADO"
        and regex_merchant.value != client.value
    ):
        merchant = regex_merchant

    # If the LLM client NIF is a helpline/toll-free number or grounded to a telecom line, drop it
    if client.status in {"ENCONTRADO", "INVALIDO"} and client.value is not None:
        if any(client.value.startswith(p) for p in ("800", "808", "707", "708")):
            client = FieldValue(value=None, status="AUSENTE")
        elif client.line_id is not None:
            line_idx = client.line_id - 1
            if 0 <= line_idx < len(doc.lines):
                line_text = doc.lines[line_idx].text.lower()
                if re.search(r"\b(?:apoio|suporte|linha|chamada|atendimento)\b", line_text):
                    client = FieldValue(value=None, status="AUSENTE")

    if client.status == "ENCONTRADO" and client.value is not None and client.value == merchant.value:
        client = FieldValue(value=None, status="AUSENTE")

    return merchant, client


def _absent_extraction(method: str) -> Extraction:
    """Build an extraction with every field absent."""
    absent = FieldValue(value=None, status="AUSENTE")
    return Extraction(
        entidade=absent,
        nif_comerciante=absent,
        nif_cliente=absent,
        data=absent,
        total=absent,
        method=cast(MethodName, method),
        elapsed_ms=0.0,
    )


def extract(
    doc: OcrDocument,
    qr: dict[str, str] | None = None,
    client: LLMClient | None = None,
) -> Extraction:
    """Extract the five required fields with a language model."""
    started = time.perf_counter()

    if client is None:
        from src.extraction.llm_client import build_client

        client = build_client()

    if not doc.lines:
        return _absent_extraction(client.provider)

    raw = client.complete(SYSTEM_PROMPT, doc.as_prompt_text(), schema=SCHEMA)
    payload = _parse_json_object(raw)
    _require_shape(payload)

    merchant = _field_from_payload(payload, "nif_comerciante", doc.lines)
    client_field = _field_from_payload(payload, "nif_cliente", doc.lines)
    merchant, client_field = _reconcile_nifs(merchant, client_field, doc, payload)

    date_field = _field_from_payload(payload, "data", doc.lines)
    if date_field.status == "AUSENTE":
        from src.extraction.regex import _extract_date

        regex_date = _extract_date(doc.lines)
        if regex_date.status == "ENCONTRADO":
            date_field = regex_date

    total_field = _field_from_payload(payload, "total", doc.lines)
    from src.extraction.regex import _extract_total

    regex_total = _extract_total(doc.lines)
    if total_field.status == "AUSENTE":
        if regex_total.status == "ENCONTRADO":
            total_field = regex_total
    elif total_field.status == "ENCONTRADO" and total_field.value is not None:
        try:
            dec_val = Decimal(total_field.value)
            # If total exceeds 500 Euros and regex found a valid decimal total, recover regex
            if dec_val > Decimal("500.00") and regex_total.status == "ENCONTRADO":
                total_field = regex_total
        except InvalidOperation:
            pass

    entity_field = _field_from_payload(payload, "entidade", doc.lines)

    return Extraction(
        entidade=entity_field,
        nif_comerciante=merchant,
        nif_cliente=client_field,
        data=date_field,
        total=total_field,
        method=cast(MethodName, client.last_effective_provider),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        raw_response=raw,
    )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to inspect the LLM extractor on one receipt."""
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.extraction.llm_client import LLMError, build_client
    from src.ocr import build_ocr_engine
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="LLM extractor preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument(
        "--engine",
        type=str,
        choices=OCR_ENGINES,
        default=None,
        help="OCR engine to use (defaults to the configured one)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=LLM_PROVIDERS,
        default="llm_local",
        help="LLM provider to query (cloud providers fall back to local)",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")
    parser.add_argument("--show-raw", action="store_true", help="Print the raw model response")

    effective_argv = list(argv) if argv is not None else (sys.argv[1:] if __name__ == "__main__" else [])
    args = parser.parse_args(effective_argv)

    settings = settings_override or get_settings()

    loaded = resolve_and_load_receipt(args.file, settings)
    if loaded is None:
        return
    target_path, raw_image = loaded

    if args.no_preprocess:
        image_to_read = raw_image
    else:
        print("Preprocessing receipt...")
        image_to_read = preprocess(raw_image, PreprocessOptions()).image

    engine = build_ocr_engine(args.engine, settings)

    print(f"Running OCR with {engine.name}...")
    doc = engine.read(image_to_read)

    client = build_client(args.provider, settings=settings)
    print(f"Querying {args.provider}...")

    try:
        extraction = extract(doc, client=client)
    except LLMError as exc:
        print(f"Language model unavailable: {exc}")
        return
    except LLMResponseError as exc:
        print(f"Unusable model response: {exc}")
        return

    print(f"\n--- LLM extraction ({extraction.method}, {extraction.elapsed_ms:.2f} ms) ---")
    for name, field in extraction.fields.items():
        print(format_extraction_field(name, field))

    if args.show_raw and extraction.raw_response is not None:
        print("\n--- Raw response ---")
        print(extraction.raw_response)


if __name__ == "__main__":
    main()
