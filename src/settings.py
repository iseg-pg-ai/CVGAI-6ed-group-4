"""Application settings loaded from environment variables and the .env file.

Centralises every environment access and filesystem path resolution so no other module
in the codebase calls os.getenv or deals with relative path quirks.

Ref: Docs/Planning/implementation_plan.md, section 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.types import OcrEngineName

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
        openai_api_key: OpenAI API key or None when disabled.
        openai_model: OpenAI model identifier.
        ocr_engine: Active OCR engine ('easyocr' or 'tesseract').
        easyocr_langs: Tuple of language codes for EasyOCR.
        tesseract_cmd: Explicit path to the tesseract executable or None for PATH lookup.
        repo_root: Absolute path to the repository root directory.
        data_dir: Absolute path to the data directory.
        receipts_dir: Absolute path to the receipts image directory.
        ground_truth_path: Absolute path to the ground truth CSV file.
        results_dir: Absolute path to the benchmark results directory.
    """

    llama_base_url: str
    llama_model: str
    llama_timeout_s: float
    gemini_api_key: str | None
    gemini_model: str
    openai_api_key: str | None
    openai_model: str
    ocr_engine: OcrEngineType
    easyocr_langs: tuple[str, ...]
    tesseract_cmd: Path | None
    repo_root: Path
    data_dir: Path
    receipts_dir: Path
    ground_truth_path: Path
    results_dir: Path


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
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=False)

    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1").strip()
    llama_model = os.getenv("LLAMA_MODEL", "local-llm").strip()
    try:
        llama_timeout_s = float(os.getenv("LLAMA_TIMEOUT_S", "120"))
    except ValueError:
        llama_timeout_s = 120.0

    gemini_api_key = _normalize_optional_str(os.getenv("GEMINI_API_KEY"))
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    openai_api_key = _normalize_optional_str(os.getenv("OPENAI_API_KEY"))
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    raw_engine = os.getenv("OCR_ENGINE", "easyocr").strip().lower()
    ocr_engine: OcrEngineType = "tesseract" if raw_engine == "tesseract" else "easyocr"

    raw_langs = os.getenv("EASYOCR_LANGS", "pt,en")
    easyocr_langs = tuple(lang.strip() for lang in raw_langs.split(",") if lang.strip()) or ("pt", "en")

    raw_tesseract_cmd = _normalize_optional_str(os.getenv("TESSERACT_CMD"))
    tesseract_cmd = Path(raw_tesseract_cmd).resolve() if raw_tesseract_cmd else None

    data_dir = _resolve_path(root, os.getenv("DATA_DIR", "data"))
    receipts_dir = _resolve_path(root, os.getenv("RECEIPTS_DIR", "data/receipts"))
    ground_truth_path = _resolve_path(root, os.getenv("GROUND_TRUTH_PATH", "data/ground_truth.csv"))
    results_dir = _resolve_path(root, os.getenv("RESULTS_DIR", "results"))

    return Settings(
        llama_base_url=llama_base_url,
        llama_model=llama_model,
        llama_timeout_s=llama_timeout_s,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        ocr_engine=ocr_engine,
        easyocr_langs=easyocr_langs,
        tesseract_cmd=tesseract_cmd,
        repo_root=root,
        data_dir=data_dir,
        receipts_dir=receipts_dir,
        ground_truth_path=ground_truth_path,
        results_dir=results_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return load_settings()
