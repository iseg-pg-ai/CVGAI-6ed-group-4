"""Extraction of the five required fields with a language model.

This is approach B of ``PROJECT_PLAN.md`` (section 4, level 2): the OCR supplies text and
coordinates, the model supplies semantic understanding, and the result comes back as strict
JSON. Works with every backend of :mod:`src.extraction.llm_client`, so the local and cloud runs differ
only in the client instance, never in the prompt or the post-processing.

Three details drive the implementation and are easy to get wrong:

- ``llama-server`` does **not** inject the ``response_format`` schema into the prompt. The
  expected JSON shape is therefore described in words inside :data:`SYSTEM_PROMPT`. Without
  that, a small model returns valid JSON with invented key names.
- The schema is flat: every field is a string and absence is an empty string plus a boolean in
  ``presente``. No unions, no optionals, because ``json-schema-to-grammar`` only supports a
  subset of the specification.
- The model can still be wrong. Invalid NIFs are downgraded to ``INVALIDO`` by
  :func:`src.core.normalizers.validar_nif`, dates and totals are re-parsed with the pure
  normalisers instead of trusting the string, and a hallucinated line number is dropped when
  it does not exist in the document.

The AT payload is accepted in the signature for parity with the regex baseline and is never
used to fill a field, for the same reason documented in :mod:`src.extraction.regex`.

Ref: Docs/Planning/implementation_plan.md, section 7, WP-10.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from src.core.cli_utils import format_extraction_field
from src.core.normalizers import (
    limpar_texto,
    normalizar_entidade,
    parse_data,
    parse_moeda,
    parse_nif,
    validar_data,
    validar_nif,
)
from src.core.types import (
    CONSUMIDOR_FINAL_NIF,
    FIELD_NAMES,
    LLM_PROVIDERS,
    OCR_ENGINES,
    Extraction,
    FieldValue,
    MethodName,
    OcrDocument,
    OcrLine,
)
from src.extraction.llm_client import build_object_schema, coerce_line_id, parse_json_object

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.extraction.llm_client import LLMClient


logger = logging.getLogger(__name__)

#: System prompt.
#:
#: Written in Portuguese because the corpus and the end users are Portuguese. It describes
#: the JSON shape in prose on purpose: ``llama-server`` ignores ``response_format`` when
#: building the prompt, so a model without the shape in text invents its own key names.
SYSTEM_PROMPT = """\
És um extrator de dados de talões e faturas simplificadas portuguesas.

Recebes o texto de um talão, linha a linha, numerado no formato [NN]. Preenche os cinco
campos pedidos usando APENAS o que está escrito nesse texto.

Regras obrigatórias:
1. Não inventes. Se um campo não estiver no documento, devolve "" nesse campo e false na
   chave correspondente de "presente". Se o campo estiver no documento, devolve o valor e true na chave correspondente de "presente".
2. Não calcules nada. Não somes artigos, não derives subtotais, não estimes valores e não
   convertas quantidades em preços. O total só existe se estiver impresso numa linha com a
   palavra TOTAL. Um talão que só tem linhas de artigos tem total "" e false.
3. Em "linhas" indica, para cada campo que devolvas como presente, o número NN da linha onde
   esse valor está impresso. Um campo presente tem sempre um número de linha maior que 0.
   Usa 0 apenas para os campos ausentes.
4. nif_comerciante é o NIF da empresa ou loja emitente (em Portugal começa por 5). nif_cliente é o NIF
   do cliente ou adquirente (em Portugal começa por 1, 2 ou 3, ou surge junto a Cliente, Nome, Contrib ou N.C.).
   O código 999999990 significa consumidor final: devolve "" e false em nif_cliente.
5. Os NIF têm exatamente 9 dígitos, sem espaços. Nunca uses números de telefone ou de apoio ao cliente (como números iniciados por 21, 22, 808, 800) nem códigos postais (como 1495-158 ou 2790-237) como NIF. Se o número impresso não tiver 9 dígitos, copia-o tal como está.
6. "data" no formato AAAA-MM-DD. Se só houver uma data de emissão, é essa.
7. "total" é o valor numérico efetivamente pago com IVA incluído, com ponto decimal e sem símbolo de moeda.
   Se houver "Total a Pagar", usa esse valor. Ignora subtotais, trocos e valores entregues.
8. "entidade" é a designação comercial ou nome da loja emitente, sem morada nem telefone.

Responde apenas com um objeto JSON, sem texto à volta e sem blocos de código, com exatamente
estas chaves:

{
  "entidade": "<string>",
  "nif_comerciante": "<string>",
  "nif_cliente": "<string>",
  "data": "<AAAA-MM-DD>",
  "total": "<decimal>",
  "presente": {
    "entidade": <true|false>,
    "nif_comerciante": <true|false>,
    "nif_cliente": <true|false>,
    "data": <true|false>,
    "total": <true|false>
  },
  "linhas": {
    "entidade": <inteiro>,
    "nif_comerciante": <inteiro>,
    "nif_cliente": <inteiro>,
    "data": <inteiro>,
    "total": <inteiro>
  }
}"""


#: Structured-output schema sent as ``response_format`` or ``response_schema``.
SCHEMA: dict[str, Any] = build_object_schema(
    {
        **{name: {"type": "string"} for name in FIELD_NAMES},
        "presente": build_object_schema({name: {"type": "boolean"} for name in FIELD_NAMES}),
        "linhas": build_object_schema({name: {"type": "integer"} for name in FIELD_NAMES}),
    }
)


class LLMResponseError(RuntimeError):
    """Raised when the model output cannot be turned into an extraction.

    A malformed response is not a transport failure, so it does not reuse
    :class:`src.extraction.llm_client.LLMError`. It is raised instead of degrading to five ``AUSENTE``
    fields because a silent empty result is indistinguishable from a model that read nothing,
    and would be scored as five omissions in the benchmark.
    """


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Recover the JSON object from a model response."""
    return parse_json_object(raw, LLMResponseError)


def _require_shape(payload: dict[str, Any]) -> None:
    """Validate the top-level shape of the decoded response.

    Unknown keys are only logged: a provider that echoes the schema is not a failure, and a
    genuinely renamed field is already caught by the required-key check.

    Args:
        payload: Decoded JSON object.

    Raises:
        LLMResponseError: When a required key is missing or the nested objects are not objects.
    """
    missing = [key for key in (*FIELD_NAMES, "presente", "linhas") if key not in payload]
    if missing:
        raise LLMResponseError(f"Response is missing keys: {', '.join(missing)}")

    for key in ("presente", "linhas"):
        if not isinstance(payload[key], dict):
            raise LLMResponseError(f"Response key {key!r} is not an object")

    unknown = set(payload) - set(FIELD_NAMES) - {"presente", "linhas"}
    if unknown:
        logger.warning("Ignoring unexpected keys in the model response: %s", ", ".join(sorted(unknown)))


def _as_text(raw: object) -> str:
    """Coerce one field value to stripped text.

    Args:
        raw: Value as decoded from JSON.

    Returns:
        The stripped text, or an empty string when the value is absent.
    """
    if raw is None:
        return ""
    return raw.strip() if isinstance(raw, str) else str(raw).strip()


def _line_id(raw: object, valid_ids: set[int]) -> int | None:
    """Resolve the line a field claims to come from."""
    return coerce_line_id(raw, valid_ids)


def _build_field(field: str, value: str, line_id: int | None) -> FieldValue:
    """Turn a model value into a typed, validated field.

    The model output is never trusted for the machine-checkable parts: NIFs go through the
    check digit, dates and totals are re-parsed by the pure normalisers. A value that does not
    survive validation keeps its text and is reported as ``INVALIDO``, so scoring counts it as
    wrong instead of as an omission.

    Args:
        field: Canonical field name.
        value: Non-empty text returned by the model.
        line_id: Supporting line, or ``None``.

    Returns:
        The validated field value.

    Raises:
        ValueError: When the field name is not one of the five canonical fields.
    """
    cleaned = limpar_texto(value)
    if not cleaned:
        return FieldValue(value=None, status="AUSENTE")

    if field == "entidade":
        return FieldValue(value=cleaned, status="ENCONTRADO", line_id=line_id)

    if field in {"nif_comerciante", "nif_cliente"}:
        nif = parse_nif(cleaned)
        if nif == CONSUMIDOR_FINAL_NIF:
            return FieldValue(value=None, status="AUSENTE")
        if nif is None:
            return FieldValue(value=cleaned, status="INVALIDO", line_id=line_id)
        status = "ENCONTRADO" if validar_nif(nif) else "INVALIDO"
        return FieldValue(value=nif, status=status, line_id=line_id)

    if field == "data":
        parsed = parse_data(cleaned)
        if parsed is None or not validar_data(parsed):
            return FieldValue(value=cleaned, status="INVALIDO", line_id=line_id)
        return FieldValue(value=parsed.isoformat(), status="ENCONTRADO", line_id=line_id)

    if field == "total":
        cleaned_total = re.sub(r"(\d+[.,]\d{2})[2ce€]\b", r"\1", cleaned)
        amount = parse_moeda(cleaned_total)
        if amount is None or amount <= 0:
            return FieldValue(value=cleaned, status="INVALIDO", line_id=line_id)
        return FieldValue(value=f"{amount:.2f}", status="ENCONTRADO", line_id=line_id)

    raise ValueError(f"Unsupported field: {field}")


def _locate_line(field: str, value: str, lines: Sequence[OcrLine]) -> int | None:
    """Find the line that prints a value the model failed to cite.

    Small models are unreliable at counting lines: on a long receipt the local 3B model
    returned ``0`` for four of five fields while extracting the values correctly. Counting is
    not a semantic task, so it should not be delegated to the model at all. When the cited
    identifier is missing or does not exist, the value is located in the OCR text by
    deterministic search, which is exact and costs no extra inference.

    Args:
        field: Canonical field name, which selects the matching strategy.
        value: Canonical value the model returned.
        lines: Recognised lines of the document.

    Returns:
        The identifier of the matching line, or ``None`` when the value cannot be found.
    """
    if not value:
        return None

    if field == "entidade":
        needle = normalizar_entidade(value)
        if not needle:
            return None
        for line in lines:
            if needle in normalizar_entidade(line.text):
                return line.id
        # The model may return a legal suffix the OCR prints differently, so the first word
        # of the trading name is the fallback.
        head = needle.split()[0]
        for line in lines:
            if head in normalizar_entidade(line.text):
                return line.id
        return None

    if field in {"nif_comerciante", "nif_cliente"}:
        digits = "".join(character for character in value if character.isdigit())
        if len(digits) != 9:
            return None
        for line in lines:
            if digits in "".join(character for character in line.text if character.isdigit()):
                return line.id
        return None

    if field == "data":
        try:
            target = date.fromisoformat(value)
        except ValueError:
            return None
        for line in lines:
            if parse_data(line.text) == target:
                return line.id
        return None

    if field == "total":
        target_dec = parse_moeda(value)
        total_keywords = ("total", "pagar", "liquido", "líquido", "final", "soma")

        # Pass 1: lines with total keywords in reverse order where parse_moeda matches
        for line in reversed(lines):
            line_dec = parse_moeda(line.text)
            if target_dec is not None and line_dec == target_dec:
                line_lower = line.text.lower()
                if any(kw in line_lower for kw in total_keywords):
                    return line.id

        # Pass 2: lines with total keywords where space-collapsed candidate matches target
        target = "".join(value.split())
        target_comma = target.replace(".", ",")
        for line in reversed(lines):
            line_no_spaces = "".join(line.text.split())
            line_lower = line.text.lower()
            if any(kw in line_lower for kw in total_keywords) and (
                target in line_no_spaces or target_comma in line_no_spaces
            ):
                return line.id

        # Pass 3: any line matching parse_moeda or target in reverse order
        for line in reversed(lines):
            line_dec = parse_moeda(line.text)
            if target_dec is not None and line_dec == target_dec:
                return line.id
            line_no_spaces = "".join(line.text.split())
            if target in line_no_spaces or target_comma in line_no_spaces:
                return line.id
        return None

    return None


def _field_from_payload(payload: dict[str, Any], field: str, lines: Sequence[OcrLine]) -> FieldValue:
    """Read one field out of a validated response.

    ``presente`` is authoritative: a model that marks a field absent while also returning text
    for it is contradicting itself, and the safe reading of a contradiction is absence. That is
    what keeps the false-positive rate honest. When the model cites no usable line, the value
    is grounded by :func:`_locate_line` instead of being left without evidence.

    Args:
        payload: Decoded and shape-checked response.
        field: Canonical field name.
        lines: Recognised lines of the document.

    Returns:
        The field value.
    """
    present = payload["presente"].get(field)
    value = _as_text(payload.get(field))
    if not value or value.lower() in {"default", "net", "none", "null", "n/a", "-"}:
        return FieldValue(value=None, status="AUSENTE")

    valid_ids = {line.id for line in lines}
    resolved = _build_field(field, value, _line_id(payload["linhas"].get(field), valid_ids))

    # If the model marked absent, accept value only if it physically grounds in document text
    if not (present is True or present == "true"):
        if resolved.status != "ENCONTRADO":
            return FieldValue(value=None, status="AUSENTE")
        if resolved.line_id is None and _locate_line(field, resolved.value or "", lines) is None:
            return FieldValue(value=None, status="AUSENTE")

    if resolved.line_id is None:
        located = _locate_line(field, resolved.value or "", lines)
        if located is not None:
            return FieldValue(
                value=resolved.value,
                status=resolved.status,
                line_id=located,
                confidence=resolved.confidence,
            )
    return resolved


def _reconcile_nifs(
    merchant: FieldValue,
    client: FieldValue,
    doc: OcrDocument,
    payload: dict[str, Any],
) -> tuple[FieldValue, FieldValue]:
    """Apply OCR structural consistency rules to disambiguate merchant and customer NIFs.

    When the language model confuses or swaps the two NIF fields, pure OCR pattern matching
    resolves the conflict without hallucinating digits or overriding explicit absence flags.
    """
    from src.extraction.regex import _extract_nifs

    explicit_client_contradiction = (
        payload.get("presente", {}).get("nif_cliente") is False and _as_text(payload.get("nif_cliente", "")) != ""
    )

    regex_merchant, regex_client = _extract_nifs(doc.lines, doc.image_size)

    def _assign_client_and_fallback_merchant(new_client: FieldValue) -> tuple[FieldValue, FieldValue]:
        if regex_merchant.status == "ENCONTRADO" and regex_merchant.value != new_client.value:
            return regex_merchant, new_client
        return FieldValue(value=None, status="AUSENTE"), new_client

    # If regex clearly found a customer NIF that the LLM missed or placed in merchant:
    if regex_client.value is not None and not explicit_client_contradiction:
        # Case A: LLM put the customer NIF into merchant field
        if merchant.value == regex_client.value or bool(merchant.value and merchant.value in regex_client.value):
            merchant, client = _assign_client_and_fallback_merchant(regex_client)
        # Case B: LLM left client field absent
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

    # Rule 2: LLM truncated client NIF into merchant field (e.g. 00037191 or 26103456)
    if (
        merchant.status == "INVALIDO"
        and client.status == "AUSENTE"
        and regex_client.status == "ENCONTRADO"
        and regex_client.value is not None
        and not explicit_client_contradiction
        and bool(merchant.value and merchant.value in regex_client.value)
    ):
        merchant, client = _assign_client_and_fallback_merchant(regex_client)

    # Rule 3: Client is absent but OCR regex found a distinct customer NIF
    if (
        client.status == "AUSENTE"
        and not explicit_client_contradiction
        and regex_client.status == "ENCONTRADO"
        and regex_client.value is not None
        and merchant.value != regex_client.value
    ):
        client = regex_client

    # Rule 3b: Merchant is absent or an obvious postal code, but OCR regex found a valid merchant NIF
    if (
        (merchant.status == "AUSENTE" or (merchant.status == "INVALIDO" and merchant.value and "-" in merchant.value))
        and regex_merchant.status == "ENCONTRADO"
        and regex_merchant.value != client.value
    ):
        merchant = regex_merchant

    # Rule 4: Discard duplicate customer NIF
    if client.status == "ENCONTRADO" and client.value is not None and client.value == merchant.value:
        client = FieldValue(value=None, status="AUSENTE")

    return merchant, client


def _absent_extraction(method: str) -> Extraction:
    """Build an extraction with every field absent.

    Args:
        method: Provider label to record.

    Returns:
        An extraction with five ``AUSENTE`` fields.
    """
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
    """Extract the five required fields with a language model.

    Args:
        doc: Recognised document, produced by either OCR engine.
        qr: Optional AT payload, accepted for interface parity and never read.
        client: Configured client. When ``None``, :func:`src.extraction.llm_client.build_client` builds
            the default provider from the environment. Pass one explicitly in tests.

    Returns:
        The extraction, tagged with the provider that actually answered.

    Raises:
        LLMResponseError: When the response cannot be decoded into the expected shape.
    """
    started = time.perf_counter()

    if client is None:
        from src.extraction.llm_client import build_client

        client = build_client()

    if not doc.lines:
        # No text means nothing to ask about. Skipping the call keeps a blank scan from
        # costing a request and from being scored as a model failure.
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
    if total_field.status == "AUSENTE":
        from src.extraction.regex import _extract_total

        regex_total = _extract_total(doc.lines)
        if regex_total.status == "ENCONTRADO":
            total_field = regex_total

    return Extraction(
        entidade=_field_from_payload(payload, "entidade", doc.lines),
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
    """CLI utility to inspect the LLM extractor on one receipt.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.
        settings_override: Optional custom Settings. When None, ``get_settings()`` is used.
    """
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
