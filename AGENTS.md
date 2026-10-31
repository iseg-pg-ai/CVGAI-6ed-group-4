# Repository Agent Guide (CVGAI-6ed-group-4)

Smart reader for Portuguese purchase receipts and simplified invoices. Hybrid pipeline: OpenCV preprocessing -> OCR (EasyOCR / Tesseract) -> structured extraction (Regex / Local LLM / Cloud LLM) -> verifiable Q&A with visual grounding -> Gradio UI.

## Environment & Toolchain

- **Manager**: `uv` only (do not use raw `pip`, `poetry`, or `conda`).
- **Python Version**: Strictly `>=3.12,<3.13` (enforced in `pyproject.toml` and `.python-version`). EasyOCR drags `torch`/`torchvision` which lacks stable wheels on >=3.14.
- **OpenCV Constraint**: Use only `opencv-python-headless`. Never install `opencv-contrib-python` or `opencv-python`; having multiple distributions claiming `cv2` corrupts the environment. `cv2.QRCodeDetector` is fully functional in headless.
- **Execution Mode**: `[tool.uv] package = false`. Run application as `python -m src.app` from root; do not package or install `src` as an editable distribution.
- **Windows Gotcha**: Set `$env:PYTHONUTF8 = "1"` before running CLI scripts to prevent `UnicodeEncodeError` on Portuguese characters. Always use `pathlib.Path`, never hardcoded backslashes or `os.sep`.

## Verification Commands

Always run checks in this order before completing work:

```bash
# 1. Formatting (line length 120)
uv run ruff format --line-length 120 src tests

# 2. Linting
uv run ruff check src tests

# 3. Type Checking
uv run pyright src tests

# 4. Test Suite (offline and fast)
uv run pytest
```

### Focused Testing Shortcuts

```bash
uv run pytest tests/test_smoke.py -q          # Fast smoke / environment verification
uv run pytest tests/test_normalizers.py -q    # Single test file
uv run pytest -k "nif" -q                     # Filter tests by expression
uv run pytest -m "not slow"                   # Skip live model/OCR suites
uv run pytest --cov=src --cov-report=term-missing
```

## Architecture & Data Contracts

- **Keystone (`src/types.py`)**: All pipeline stages exchange only classes from `src/types.py` (`BBox`, `OcrLine`, `OcrDocument`, `FieldValue`, `Extraction`, `GroundTruthRecord`, `MatchResult`, `Answer`). Do not modify contracts in `types.py` without coordinating changes.
- **Canonical Fields**: The 5 required extraction fields are `entidade`, `nif_comerciante`, `nif_cliente`, `data`, and `total`.
- **Absent vs Invalid**:
  - In `GroundTruthRecord`, absent fields are `None` (empty CSV field in `data/ground_truth.csv`), never `0`, `"-"`, or `"N/A"`.
  - In `FieldValue`, status must be one of `"ENCONTRADO"`, `"AUSENTE"`, or `"INVALIDO"`.
- **NIF Validation**: Portuguese NIFs must be 9 digits and pass modulo-11 check digit verification. Invalid NIFs are classified as `"INVALIDO"`.
- **QR Code Reader (`src/qr_reader.py`)**: Parses Portuguese Tax Authority (AT) format (`*` separated `KEY:VALUE` pairs). Used for cross-validation only; never rely on QR alone because payment terminal slips have no QR.
- **LLM Prompting Gotcha**: `llama-server` does not inject `response_format` schemas into the prompt text automatically. Prompts must explicitly describe the expected JSON schema. Keep schemas flat without union types to avoid breaking `llama.cpp`'s `json-schema-to-grammar`.

## Testing Rules

1. Tests must run completely offline: no network requests, no GPU requirements, no external API keys, and no heavy live model weights.
2. OCR tests must use synthetic images or mocked `OcrDocument` objects, never loading real EasyOCR weights in standard unit tests.
3. Temporary artifacts must be written to pytest's `tmp_path` fixture, never committed to `data/` or root directories.

## Style & Workflow

- **No Emojis**: Do not use emojis anywhere (code, docstrings, commits, PRs, comments, or UI text).
- **Docstrings**: Google-style docstrings in English for all public functions and classes.
- **Git**: Trunk-based development with short-lived feature branches (`feat/wp-*`, `fix/*`). Target diff size < 200 lines per work package.
