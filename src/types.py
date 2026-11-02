"""Shared data contracts for the whole pipeline.

This module is the coordination keystone of the project: every stage (preprocessing, OCR,
extraction, question answering, scoring) exchanges only the types defined here. Changing a
field or a name in this file is a contract change and must be announced to the team.

Ref: Docs/Planning/implementation_plan.md, section 3.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal, Protocol

import numpy as np

#: Outcome of trying to read one field from a document.
Status = Literal["ENCONTRADO", "AUSENTE", "INVALIDO"]

#: Canonical extraction field names and their runtime order.
FieldName = Literal["entidade", "nif_comerciante", "nif_cliente", "data", "total"]
FIELD_NAMES: tuple[FieldName, ...] = (
    "entidade",
    "nif_comerciante",
    "nif_cliente",
    "data",
    "total",
)

#: Receipt physical condition groups in the corpus.
GroupName = Literal["direito", "amarrotado"]
GROUPS: tuple[GroupName, ...] = ("direito", "amarrotado")

#: Supported OCR engine names.
OcrEngineName = Literal["easyocr", "tesseract"]
OCR_ENGINES: tuple[OcrEngineName, ...] = ("easyocr", "tesseract")

#: Supported LLM provider backend identifiers.
ProviderName = Literal["llm_local", "llm_gemini", "llm_openai"]
LLM_PROVIDERS: tuple[ProviderName, ...] = ("llm_local", "llm_gemini", "llm_openai")

#: Supported extraction method identifiers.
MethodName = Literal["regex", "llm_local", "llm_gemini", "llm_openai"]
EXTRACTION_METHODS: tuple[MethodName, ...] = ("regex", "llm_local", "llm_gemini", "llm_openai")

#: AT sentinel printed in the B field of a final-consumer invoice (passes NIF check digit).
CONSUMIDOR_FINAL_NIF = "999999990"

#: Earliest issue date considered plausible on a receipt.
EARLIEST_PLAUSIBLE_DATE = date(2000, 1, 1)

#: Supported image file extensions when scanning receipt directories.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box in pixels, in the reference frame of the processed image.

    Attributes:
        x1: Left edge, inclusive.
        y1: Top edge, inclusive.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width of the box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the box in pixels."""
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class OcrLine:
    """A recognised text line, with a stable identifier and its geometry.

    Attributes:
        id: One-based line number. The LLM cites this identifier to ground an answer, so it
            must stay stable for the lifetime of the document.
        text: Text of the line, with single spaces between words.
        confidence: Engine confidence in ``[0, 1]``. Use ``0.0`` when unknown.
        bbox: Box of the line in the reference frame of the processed image.
    """

    id: int
    text: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True, slots=True)
class OcrDocument:
    """Full result of one OCR engine over one image.

    Attributes:
        lines: Recognised lines in reading order.
        engine: Name of the engine that produced the document, e.g. ``"easyocr"``.
        elapsed_ms: Wall-clock time of the recognition, in milliseconds.
        image_size: Size of the processed image as ``(width, height)``.
    """

    lines: list[OcrLine]
    engine: str
    elapsed_ms: float
    image_size: tuple[int, int]

    def as_prompt_text(self) -> str:
        """Render the lines as a numbered block suitable for an LLM prompt.

        Each line is emitted as ``[NN] text`` so the model can cite ``NN`` as evidence.

        Returns:
            One line per recognised line, separated by newlines.
        """
        width = max(2, len(str(len(self.lines))))
        return "\n".join(f"[{line.id:0{width}d}] {line.text}" for line in self.lines)


class OcrEngine(Protocol):
    """Interface every OCR engine must satisfy.

    Implementations live in ``ocr_engine.py`` (EasyOCR) and ``ocr_tesseract.py`` (Tesseract).
    Because the two engines have incompatible runtime costs, only one of them is active per
    request; the caller selects it explicitly.
    """

    name: str

    def read(self, image: np.ndarray) -> OcrDocument:
        """Recognise the text of one image.

        Args:
            image: BGR image, already preprocessed and aligned.

        Returns:
            The recognised document, with sequential line identifiers.
        """
        ...


@dataclass(frozen=True, slots=True)
class FieldValue:
    """Extracted value of one field, with its status and supporting evidence.

    Attributes:
        value: The extracted value, or ``None`` when nothing was read.
        status: Whether the field was found, absent from the document, or found but
            semantically invalid (a NIF that fails the check digit, for instance).
        line_id: Identifier of the OCR line that supports the value, when known.
        confidence: Extractor confidence in ``[0, 1]``, when known.
    """

    value: str | None
    status: Status
    line_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Extraction:
    """The five fields required by the assignment, plus run metadata.

    Attributes:
        entidade: Trading name of the merchant.
        nif_comerciante: Merchant taxpayer number.
        nif_cliente: Customer taxpayer number.
        data: Date of issue.
        total: Total amount paid.
        method: Which extractor produced this result, e.g. ``"regex"``, ``"llm_local"``,
            ``"llm_gemini"`` or ``"llm_openai"``.
        elapsed_ms: Wall-clock time of the extraction, in milliseconds.
        raw_response: Raw model response, kept for debugging and for the report appendix.
    """

    entidade: FieldValue
    nif_comerciante: FieldValue
    nif_cliente: FieldValue
    data: FieldValue
    total: FieldValue
    method: str
    elapsed_ms: float
    raw_response: str | None = None

    @property
    def fields(self) -> dict[str, FieldValue]:
        """The five fields keyed by CSV column name, in canonical order."""
        return {
            "entidade": self.entidade,
            "nif_comerciante": self.nif_comerciante,
            "nif_cliente": self.nif_cliente,
            "data": self.data,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class GroundTruthRecord:
    """One validated, typed row of ``data/ground_truth.csv``.

    An empty CSV cell becomes ``None``. An absent field is never encoded as ``0``, ``"N/A"``
    or ``"-"``: the distinction between "absent" and "unreadable" is what makes the
    false-positive rate meaningful.

    Attributes:
        filename: Image file name inside ``data/receipts/``, including the extension.
        grupo: ``"direito"`` for a flat receipt, ``"amarrotado"`` for a crumpled one.
        entidade: Trading name as printed, or ``None`` when absent.
        nif_comerciante: Nine-digit merchant NIF, or ``None`` when absent.
        nif_cliente: Nine-digit customer NIF, or ``None`` when absent.
        data: Issue date, or ``None`` when absent.
        total: Total amount, or ``None`` when absent.
    """

    filename: str
    grupo: GroupName
    entidade: str | None
    nif_comerciante: str | None
    nif_cliente: str | None
    data: date | None
    total: Decimal | None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of comparing one field between a prediction and the reference truth.

    Attributes:
        filename: Image the result refers to.
        grupo: ``"direito"`` or ``"amarrotado"``.
        campo: Field name, matching the CSV column.
        esperado: Reference value, already normalised to a comparable string.
        obtido: Predicted value, already normalised to a comparable string.
        acertou: Whether the two values are equivalent under the field's rule.
        falso_positivo: The document had no value and the system invented one.
        omissao: The document had a value and the system returned nothing.
    """

    filename: str
    grupo: str
    campo: str
    esperado: str | None
    obtido: str | None
    acertou: bool
    falso_positivo: bool
    omissao: bool


@dataclass(frozen=True, slots=True)
class Answer:
    """Assistant answer, with its presence status and the evidence that supports it.

    The ``status`` field is what allows the application to say "this document does not
    contain that information" instead of hallucinating an answer.

    Attributes:
        question: The question as asked.
        answer: The natural-language answer, in European Portuguese.
        status: ``"ENCONTRADO"`` when the document supports the answer,
            ``"AUSENTE NO DOCUMENTO"`` when it does not.
        line_ids: Identifiers of the OCR lines that ground the answer, for highlighting.
        provider: Which model answered, e.g. ``"llm_local"`` or ``"llm_gemini"``.
    """

    question: str
    answer: str
    status: Literal["ENCONTRADO", "AUSENTE NO DOCUMENTO"]
    line_ids: list[int]
    provider: str
