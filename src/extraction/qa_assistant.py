"""Evidence-grounded question answering assistant for receipts and simplified invoices.

Answers natural language questions about a recognised receipt using the OCR text and spatial
line identifiers, returning a structured :class:`src.core.types.Answer` with presence status and
supporting line IDs for visual highlighting.

Ref: Docs/implementation_plan.md, section 7, WP-12.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from src.core.types import (
    ANSWER_STATUSES,
    LLM_PROVIDERS,
    OCR_ENGINES,
    Answer,
    AnswerStatus,
    OcrDocument,
    OcrLine,
    ProviderName,
)
from src.extraction.llm_client import build_object_schema, coerce_line_id, parse_json_object

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.extraction.llm_client import LLMClient


logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """\
Es um assistente inteligente e preciso para consulta de talões e faturas simplificadas portuguesas.

Recebes o texto de um talão, linha a linha, numerado no formato [NN], juntamente com uma pergunta do utilizador.
Responde à pergunta utilizando APENAS a informação presente no texto do documento.

Regras obrigatórias:
1. Se a informação necessária para responder à pergunta estiver presente no documento, responde em português europeu de forma clara e concisa, define o status como "ENCONTRADO", e indica em "linhas" o número ou números das linhas [NN] que fundamentam a resposta.
2. Se a informação não estiver presente no documento, responde educadamente que a informação está ausente, define o status como "AUSENTE NO DOCUMENTO", e indica uma lista de linhas vazia ([]).
3. Não inventes nem extrapoles factos que não constem explicitamente no texto.
4. "linhas" deve conter apenas inteiros correspondentes aos números das linhas [NN] presentes no documento.

Responde apenas com um objeto JSON válido, sem texto à volta e sem blocos de código, com exatamente estas chaves:

{
  "answer": "<resposta em português europeu>",
  "status": "<ENCONTRADO|AUSENTE NO DOCUMENTO>",
  "linhas": [<inteiro>, ...]
}"""


QA_SCHEMA: dict[str, Any] = build_object_schema(
    {
        "answer": {"type": "string"},
        "status": {"type": "string", "enum": list(ANSWER_STATUSES)},
        "linhas": {
            "type": "array",
            "items": {"type": "integer"},
        },
    }
)


class QAError(RuntimeError):
    """Base exception for all Q&A assistant failures."""


class QAResponseError(QAError):
    """Raised when the model output cannot be parsed into a valid Q&A response."""


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Recover the JSON object from a model response."""
    return parse_json_object(raw, QAResponseError)


def _require_shape(payload: dict[str, Any]) -> None:
    """Validate top-level shape of the decoded Q&A response."""
    missing = [key for key in ("answer", "status", "linhas") if key not in payload]
    if missing:
        raise QAResponseError(f"Response is missing keys: {', '.join(missing)}")
    if not isinstance(payload["linhas"], list):
        raise QAResponseError("Response key 'linhas' is not a list")


def _filter_line_ids(raw_lines: list[Any], valid_ids: set[int]) -> list[int]:
    """Filter and validate line identifiers returned by the model."""
    result: list[int] = []
    for item in raw_lines:
        line_num = coerce_line_id(item, valid_ids)
        if line_num is not None and line_num not in result:
            result.append(line_num)
    return result


def _locate_supporting_lines(answer: str, question: str, lines: Sequence[OcrLine]) -> list[int]:
    """Locate lines in the document that match keywords in the answer or question."""
    found: list[int] = []
    stopwords = frozenset(
        {
            "qual",
            "quais",
            "quanto",
            "quantos",
            "onde",
            "como",
            "quem",
            "para",
            "com",
            "loja",
            "recibo",
            "talao",
            "talão",
            "fatura",
            "factura",
            "documento",
            "este",
            "esta",
            "esse",
            "essa",
            "isso",
            "pelo",
            "pela",
            "sobre",
            "qualquer",
            "valor",
            "total",
            "compra",
            "tem",
            "ha",
            "há",
        }
    )

    def _extract_tokens(text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) >= 3 and w not in stopwords]

    answer_tokens = _extract_tokens(answer)
    question_tokens = _extract_tokens(question)

    if answer_tokens:
        for line in lines:
            line_lower = line.text.lower()
            if any(tok in line_lower for tok in answer_tokens) and line.id not in found:
                found.append(line.id)

    if not found and question_tokens:
        for line in lines:
            line_lower = line.text.lower()
            if any(tok in line_lower for tok in question_tokens) and line.id not in found:
                found.append(line.id)

    return found[:3]


def ask(
    question: str,
    doc: OcrDocument,
    client: LLMClient | None = None,
) -> Answer:
    """Ask a question about a recognised receipt and return an evidence-grounded answer.

    Args:
        question: Natural language question in Portuguese.
        doc: Recognised OCR document.
        client: Configured LLM client. When None, build_client() builds the default provider.

    Returns:
        Structured Answer with presence status, natural language text, and supporting line IDs.

    Raises:
        QAResponseError: When the model response cannot be parsed.
    """
    if client is None:
        from src.extraction.llm_client import build_client

        client = build_client()

    if not doc.lines:
        return Answer(
            question=question,
            answer="O documento nao contem texto reconhecido para responder à pergunta.",
            status="AUSENTE NO DOCUMENTO",
            line_ids=[],
            provider=cast(ProviderName, client.provider),
        )

    user_prompt = f"Pergunta: {question}\n\nDocumento:\n{doc.as_prompt_text()}"
    raw = client.complete(QA_SYSTEM_PROMPT, user_prompt, schema=QA_SCHEMA)
    payload = _parse_json_object(raw)
    _require_shape(payload)

    answer_text = str(payload.get("answer", "")).strip()
    status_raw = str(payload.get("status", "")).strip().upper()
    valid_ids = {line.id for line in doc.lines}
    raw_linhas = payload.get("linhas", [])
    filtered_ids = _filter_line_ids(raw_linhas, valid_ids)

    is_found = status_raw == "ENCONTRADO"
    if is_found and not filtered_ids:
        located = _locate_supporting_lines(answer_text, question, doc.lines)
        if located:
            filtered_ids = located
        else:
            is_found = False

    status: AnswerStatus = "ENCONTRADO" if is_found and filtered_ids else "AUSENTE NO DOCUMENTO"
    if status == "AUSENTE NO DOCUMENTO" and is_found:
        answer_text = "A informação solicitada não foi encontrada ou não pôde ser comprovada no documento."

    return Answer(
        question=question,
        answer=answer_text,
        status=status,
        line_ids=filtered_ids,
        provider=cast(ProviderName, client.last_effective_provider),
    )


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI utility to test Q&A assistant on a receipt.

    Args:
        argv: Optional command-line arguments. Defaults to sys.argv[1:].
        settings_override: Optional custom Settings. When None, get_settings() is used.
    """
    from src.core.cli_utils import resolve_and_load_receipt
    from src.core.settings import get_settings
    from src.extraction.llm_client import LLMError, build_client
    from src.ocr import build_ocr_engine
    from src.vision.preprocessing import PreprocessOptions, preprocess

    parser = argparse.ArgumentParser(description="Evidence-grounded Q&A assistant preview")
    parser.add_argument("--file", "-f", type=str, default=None, help="Receipt image path")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="Qual e o valor total do talão?",
        help="Question to ask",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=OCR_ENGINES,
        default=None,
        help="OCR engine to use",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=LLM_PROVIDERS,
        default="llm_local",
        help="LLM provider to query",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Skip OpenCV preprocessing")

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
    print(f"Querying {args.provider} with question: {args.question!r}...")

    try:
        answer = ask(args.question, doc, client=client)
    except LLMError as exc:
        print(f"Language model unavailable: {exc}")
        return
    except QAError as exc:
        print(f"Q&A error: {exc}")
        return

    print(f"\n--- Q&A Result ({answer.provider}) ---")
    print(f"Question: {answer.question}")
    print(f"Status:   {answer.status}")
    print(f"Answer:   {answer.answer}")
    print(f"Lines:    {answer.line_ids}")


if __name__ == "__main__":
    main()
