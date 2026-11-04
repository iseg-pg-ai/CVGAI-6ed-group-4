"""Application settings loaded from environment variables and the .env file.

Centralises every environment access and filesystem path resolution so no other module
in the codebase calls os.getenv or deals with relative path quirks.

Ref: Docs/implementation_plan.md, section 2.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from src.core.types import OCR_ENGINES, OcrEngineName

logger = logging.getLogger(__name__)

OcrEngineType = OcrEngineName


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve a path against repo_root if it is not already absolute."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


class Settings(BaseSettings):
    """Immutable application settings container.

    Attributes:
        llama_base_url: Base URL of the running llama-server instance.
        llama_model: Model name advertised by llama-server.
        llama_timeout_s: Timeout in seconds for local model inferences.
        gemini_api_key: Google Gemini API key or None when disabled.
        gemini_model: Gemini model identifier.
        ocr_engine: Active OCR engine ('easyocr' or 'tesseract').
        easyocr_langs: Tuple of language codes for EasyOCR.
        tesseract_cmd: Explicit path to the tesseract executable or None for PATH lookup.
        repo_root: Absolute path to the repository root directory.
        data_dir: Absolute path to the data directory.
        receipts_dir: Absolute path to the receipts image directory.
        ground_truth_path: Absolute path to the ground truth CSV file.
        results_dir: Absolute path to the benchmark results directory.
        max_side: Maximum image dimension for preprocessing downscale.
        clahe_clip_limit: Default CLAHE contrast limit.
        llama_port: Default local llama-server port.
        llama_ngl: Default GPU layers offloaded in local llama-server.
        llama_ctx: Context size in tokens for local llama-server.
        llama_model_path: Path to local GGUF model file or default.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="ignore",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "local-llm"
    llama_timeout_s: float = 120.0
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    ocr_engine: OcrEngineType = "easyocr"
    easyocr_langs: Annotated[tuple[str, ...], NoDecode] = ("pt", "en")
    tesseract_cmd: Path | None = None

    data_dir: Path = Path("data")
    receipts_dir: Path = Path("data/receipts")
    ground_truth_path: Path = Path("data/ground_truth.csv")
    results_dir: Path = Path("results")

    max_side: int = Field(
        default=1280,
        validation_alias=AliasChoices("PREPROCESS_MAX_SIDE", "max_side"),
    )
    clahe_clip_limit: float = Field(
        default=2.0,
        validation_alias=AliasChoices("PREPROCESS_CLAHE_CLIP", "clahe_clip_limit"),
    )
    llama_port: int = Field(
        default=8080,
        validation_alias=AliasChoices("LLAMA_PORT", "PORT", "llama_port"),
    )
    llama_ngl: int = Field(
        default=0,
        validation_alias=AliasChoices("LLAMA_NGL", "NGL", "llama_ngl"),
    )
    llama_ctx: int = Field(
        default=8192,
        validation_alias=AliasChoices("LLAMA_CTX", "CTX", "llama_ctx"),
    )
    llama_model_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LLAMA_MODEL_PATH", "MODEL", "llama_model_path"),
    )

    @field_validator("llama_base_url", "llama_model", "gemini_model", mode="before")
    @classmethod
    def _strip_strings(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _normalize_optional_key(cls, v: Any) -> str | None:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator("tesseract_cmd", mode="before")
    @classmethod
    def _normalize_tesseract_cmd(cls, v: Any) -> Path | None:
        if v is None:
            return None
        if isinstance(v, Path):
            return v.resolve()
        stripped = str(v).strip()
        return Path(stripped).resolve() if stripped else None

    @field_validator("ocr_engine", mode="before")
    @classmethod
    def _validate_ocr_engine(cls, v: Any) -> OcrEngineType:
        raw = str(v).strip().lower() if v is not None else "easyocr"
        if raw in OCR_ENGINES:
            return raw  # type: ignore[return-value]
        logger.warning("Unrecognised OCR_ENGINE '%s'; defaulting to 'easyocr'.", raw)
        return "easyocr"

    @field_validator("easyocr_langs", mode="before")
    @classmethod
    def _parse_langs(cls, v: Any) -> tuple[str, ...]:
        if isinstance(v, str):
            parsed = tuple(lang.strip() for lang in v.split(",") if lang.strip())
            return parsed or ("pt", "en")
        if isinstance(v, (list, tuple)):
            return tuple(str(x).strip() for x in v if str(x).strip()) or ("pt", "en")
        return ("pt", "en")

    @field_validator("llama_timeout_s", "clahe_clip_limit", mode="before")
    @classmethod
    def _safe_float(cls, v: Any, info: ValidationInfo) -> float:
        defaults = {"llama_timeout_s": 120.0, "clahe_clip_limit": 2.0}
        try:
            return float(v)
        except (ValueError, TypeError):
            return defaults[info.field_name or "llama_timeout_s"]

    @field_validator("max_side", "llama_port", "llama_ngl", "llama_ctx", mode="before")
    @classmethod
    def _safe_int(cls, v: Any, info: ValidationInfo) -> int:
        defaults = {"max_side": 1280, "llama_port": 8080, "llama_ngl": 0, "llama_ctx": 8192}
        try:
            return int(v)
        except (ValueError, TypeError):
            return defaults[info.field_name or "max_side"]

    @model_validator(mode="after")
    def _resolve_repo_paths(self) -> Settings:
        root = self.repo_root.resolve()
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "data_dir", _resolve_path(root, self.data_dir))
        object.__setattr__(self, "receipts_dir", _resolve_path(root, self.receipts_dir))
        object.__setattr__(self, "ground_truth_path", _resolve_path(root, self.ground_truth_path))
        object.__setattr__(self, "results_dir", _resolve_path(root, self.results_dir))
        if self.llama_model_path is not None:
            object.__setattr__(self, "llama_model_path", _resolve_path(root, self.llama_model_path))
        else:
            object.__setattr__(
                self,
                "llama_model_path",
                (root / "models" / "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf").resolve(),
            )
        return self


class _DataclassFieldProxy:
    """Minimal proxy satisfying dataclasses.replace() contract for backwards compatibility."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._field_type = getattr(dataclasses, "_FIELD", "FIELD")
        self.init = True


# Expose __dataclass_fields__ so stdlib dataclasses.replace(settings, ...) continues working
Settings.__dataclass_fields__ = {name: _DataclassFieldProxy(name) for name in Settings.model_fields}


def load_settings(repo_root: Path | None = None) -> Settings:
    """Load settings from environment variables and the repository's .env file.

    Args:
        repo_root: Optional repository root path. When omitted, deduced from the location
            of this file.

    Returns:
        A validated, immutable Settings instance.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    env_file = root / ".env"
    return Settings(
        repo_root=root,
        _env_file=env_file if env_file.is_file() else None,  # type: ignore[call-arg]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return load_settings()
