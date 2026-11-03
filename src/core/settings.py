"""Application settings loaded from environment variables and the .env file.

Centralises every environment access and filesystem path resolution so no other module
in the codebase calls os.getenv or deals with relative path quirks.

Ref: Docs/Planning/implementation_plan.md, section 2.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

from src.core.types import OCR_ENGINES, OcrEngineName

logger = logging.getLogger(__name__)

OcrEngineType = OcrEngineName


@dataclass(frozen=True, slots=True)
class Settings:
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
    """

    llama_base_url: str
    llama_model: str
    llama_timeout_s: float
    gemini_api_key: str | None
    gemini_model: str
    ocr_engine: OcrEngineType
    easyocr_langs: tuple[str, ...]
    tesseract_cmd: Path | None
    repo_root: Path
    data_dir: Path
    receipts_dir: Path
    ground_truth_path: Path
    results_dir: Path
    max_side: int = 1280
    clahe_clip_limit: float = 2.0
    llama_port: int = 8080
    llama_ngl: int = 0


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    """Resolve a path string against repo_root if it is not already absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _normalize_optional_str(value: str | None) -> str | None:
    """Return stripped string or None if empty or unset."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


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
    file_vars: dict[str, str | None] = dotenv_values(dotenv_path=env_file) if env_file.is_file() else {}

    def _get_val(key: str, default: str = "") -> str:
        env_val = os.getenv(key)
        if env_val is not None:
            return env_val
        file_val = file_vars.get(key)
        if file_val is not None:
            return file_val
        return default

    llama_base_url = _get_val("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1").strip()
    llama_model = _get_val("LLAMA_MODEL", "local-llm").strip()
    try:
        llama_timeout_s = float(_get_val("LLAMA_TIMEOUT_S", "120"))
    except ValueError:
        llama_timeout_s = 120.0

    gemini_api_key = _normalize_optional_str(_get_val("GEMINI_API_KEY", "") or None)
    gemini_model = _get_val("GEMINI_MODEL", "gemini-3.5-flash").strip()

    raw_engine = _get_val("OCR_ENGINE", "easyocr").strip().lower()
    if raw_engine in OCR_ENGINES:
        ocr_engine: OcrEngineType = raw_engine  # type: ignore[assignment]
    else:
        logger.warning("Unrecognised OCR_ENGINE '%s'; defaulting to 'easyocr'.", raw_engine)
        ocr_engine = "easyocr"

    raw_langs = _get_val("EASYOCR_LANGS", "pt,en")
    easyocr_langs = tuple(lang.strip() for lang in raw_langs.split(",") if lang.strip()) or ("pt", "en")

    raw_tesseract_cmd = _normalize_optional_str(_get_val("TESSERACT_CMD", "") or None)
    tesseract_cmd = Path(raw_tesseract_cmd).resolve() if raw_tesseract_cmd else None

    data_dir = _resolve_path(root, _get_val("DATA_DIR", "data"))
    receipts_dir = _resolve_path(root, _get_val("RECEIPTS_DIR", "data/receipts"))
    ground_truth_path = _resolve_path(root, _get_val("GROUND_TRUTH_PATH", "data/ground_truth.csv"))
    results_dir = _resolve_path(root, _get_val("RESULTS_DIR", "results"))

    try:
        max_side = int(_get_val("PREPROCESS_MAX_SIDE", "1280"))
    except ValueError:
        max_side = 1280

    try:
        clahe_clip_limit = float(_get_val("PREPROCESS_CLAHE_CLIP", "2.0"))
    except ValueError:
        clahe_clip_limit = 2.0

    try:
        llama_port = int(_get_val("LLAMA_PORT", "8080"))
    except ValueError:
        llama_port = 8080

    try:
        llama_ngl = int(_get_val("LLAMA_NGL", "0"))
    except ValueError:
        llama_ngl = 0

    return Settings(
        llama_base_url=llama_base_url,
        llama_model=llama_model,
        llama_timeout_s=llama_timeout_s,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        ocr_engine=ocr_engine,
        easyocr_langs=easyocr_langs,
        tesseract_cmd=tesseract_cmd,
        repo_root=root,
        data_dir=data_dir,
        receipts_dir=receipts_dir,
        ground_truth_path=ground_truth_path,
        results_dir=results_dir,
        max_side=max_side,
        clahe_clip_limit=clahe_clip_limit,
        llama_port=llama_port,
        llama_ngl=llama_ngl,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return load_settings()
