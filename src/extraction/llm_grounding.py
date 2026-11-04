"""Field validation, reverse OCR line grounding, and response shape verification.

Ref: Docs/implementation_plan.md, section 7, WP-10.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from src.core.normalizers import (
    limpar_texto,
    normalizar_entidade,
    parse_data,
    parse_moeda,
    parse_nif,
    validar_data,
    validar_nif,
)
from src.core.types import CONSUMIDOR_FINAL_NIF, FIELD_NAMES, FieldValue, OcrLine
from src.extraction.llm_client import coerce_line_id
from src.extraction.llm_prompts import LLMResponseError

logger = logging.getLogger(__name__)


def require_shape(payload: dict[str, Any]) -> None:
    """Validate the top-level shape of the decoded response.

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


_require_shape = require_shape


def as_text(raw: object) -> str:
    """Coerce one field value to stripped text."""
    if raw is None:
        return ""
    return raw.strip() if isinstance(raw, str) else str(raw).strip()


_as_text = as_text


def line_id_from_raw(raw: object, valid_ids: set[int]) -> int | None:
    """Resolve the line a field claims to come from."""
    return coerce_line_id(raw, valid_ids)


_line_id = line_id_from_raw


def build_field(field: str, value: str, line_id: int | None) -> FieldValue:
    """Turn a model value into a typed, validated field."""
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


_build_field = build_field


def locate_line(field: str, value: str, lines: Sequence[OcrLine]) -> int | None:
    """Find the line that prints a value the model failed to cite."""
    if not value:
        return None

    if field == "entidade":
        needle = normalizar_entidade(value)
        if not needle:
            return None
        for line in lines:
            if needle in normalizar_entidade(line.text):
                return line.id
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


_locate_line = locate_line


def field_from_payload(payload: dict[str, Any], field: str, lines: Sequence[OcrLine]) -> FieldValue:
    """Read one field out of a validated response and ground its line ID."""
    present = payload["presente"].get(field)
    value = as_text(payload.get(field))
    if not value or value.lower() in {"default", "net", "none", "null", "n/a", "-"}:
        return FieldValue(value=None, status="AUSENTE")

    valid_ids = {line.id for line in lines}
    resolved = build_field(field, value, line_id_from_raw(payload["linhas"].get(field), valid_ids))

    if not (present is True or present == "true"):
        if resolved.status != "ENCONTRADO":
            return FieldValue(value=None, status="AUSENTE")
        if resolved.line_id is None and locate_line(field, resolved.value or "", lines) is None:
            return FieldValue(value=None, status="AUSENTE")

    if resolved.line_id is None:
        located = locate_line(field, resolved.value or "", lines)
        if located is not None:
            return FieldValue(
                value=resolved.value,
                status=resolved.status,
                line_id=located,
                confidence=resolved.confidence,
            )
    return resolved


_field_from_payload = field_from_payload
