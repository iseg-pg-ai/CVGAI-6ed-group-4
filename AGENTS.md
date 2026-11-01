# Repository Agent Guide (CVGAI-6ed-group-4)

Smart reader for Portuguese purchase receipts and simplified invoices. Hybrid pipeline: OpenCV preprocessing -> OCR (EasyOCR / Tesseract) -> structured extraction (Regex / Local LLM / Cloud LLM) -> verifiable Q&A with visual grounding -> Gradio UI.

## Current State

Closed work packages: WP-00 (skeleton), WP-01 (data contracts), WP-04 (preprocessing), WP-05 (EasyOCR), WP-07 (AT QR reader), WP-11 (visual evidence). `src/settings.py` was added outside the numbering as the single environment and path layer.

- Implemented: `src/types.py`, `src/settings.py`, `src/preprocessing.py`, `src/ocr_engine.py`, `src/qr_reader.py`, `src/evidence.py`.
- Not implemented yet: `normalizers.py`, `golden_set.py`, `ocr_tesseract.py`, `extractor_regex.py`, `llm_client.py`, `extractor_llm.py`, `qa_assistant.py`, `scoring.py`, `evaluate.py`, `dataset_stats.py`, `app.py`, `scripts/`. Do not describe them as available, in docs or in code.
- `python -m src.app` does not run yet (Gradio is WP-16). Only the per-module CLI previews exist.
- Corpus: `data/receipts/` holds 9 images; `data/ground_truth.csv` has only the header (WP-17 is open).
- Suite: 63 tests, offline, about 0.3 s. One `@pytest.mark.slow` live-EasyOCR test is deselected by default.

Each module docstring carries a `Ref:` line pointing at the section of `Docs/Planning/implementation_plan.md` that specifies it. Keep that convention for new modules.

## Environment & Toolchain

- **Manager**: `uv` only (do not use raw `pip`, `poetry`, or `conda`).
- **Python Version**: Strictly `>=3.12,<3.13` (enforced in `pyproject.toml` and `.python-version`). EasyOCR drags `torch`/`torchvision` which lacks stable wheels on >=3.14.
- **OpenCV Constraint**: Use only `opencv-python-headless`. Never install `opencv-contrib-python` or `opencv-python`; having multiple distributions claiming `cv2` corrupts the environment. `cv2.QRCodeDetector` is fully functional in headless (verified on OpenCV 5).
- **Execution Mode**: `[tool.uv] package = false`. Run modules as `python -m src.<module>` from the repository root; never install `src` as an editable distribution.
- **Windows Gotcha**: Set `$env:PYTHONUTF8 = "1"` before running CLI scripts to prevent `UnicodeEncodeError` on Portuguese characters. Always use `pathlib.Path`, never hardcoded backslashes or `os.sep`.

## Verification Commands

Always run checks in this order before completing work:

```bash
# 1. Formatting (line length 120; also set in pyproject.toml)
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
uv run pytest tests/test_evidence.py -q        # Single test file
uv run pytest -k "nif" -q                      # Filter tests by expression
uv run pytest -m slow                          # Opt-in live EasyOCR test (needs weights)
uv run pytest --cov=src --cov-report=term-missing
```

`pyproject.toml` sets `addopts = "-q --strict-markers -m 'not slow'"`, so `slow` tests never run by default and an unregistered marker fails the suite.

## Architecture & Data Contracts

- **Keystone (`src/types.py`)**: All pipeline stages exchange only classes from `src/types.py` (`BBox`, `OcrLine`, `OcrDocument`, `OcrEngine`, `FieldValue`, `Extraction`, `GroundTruthRecord`, `MatchResult`, `Answer`). Do not modify contracts in `types.py` without coordinating changes.
- **Bounding boxes**: `BBox(x1, y1, x2, y2)` is in pixels, in the reference frame of the preprocessed image, with `x2`/`y2` exclusive. Every OCR engine must clamp coordinates to the image.
- **Settings (`src/settings.py`)**: The only module allowed to read environment variables or resolve repository paths. `load_settings(repo_root=...)` builds an isolated instance for a temporary tree; `get_settings()` is `lru_cache`d, so tests that change the environment must use `load_settings`, never the cached getter.
- **CLI convention**: Every runnable module exposes `main(argv: Sequence[str] | None = None, settings_override: Settings | None = None)` plus the `if __name__ == "__main__": main()` guard. Heavy imports (Pillow, EasyOCR, `src.settings`) are deferred to `main` or to the engine constructor, so importing a module stays cheap for tests.
- **Image validation**: Every public function that takes an image raises `ValueError("Input image must not be empty or zero-sized.")` on empty input and returns a new array instead of mutating the caller's one.
- **Canonical Fields**: The 5 required extraction fields are `entidade`, `nif_comerciante`, `nif_cliente`, `data`, and `total`.
- **Absent vs Invalid**:
  - In `GroundTruthRecord`, absent fields are `None` (empty CSV field in `data/ground_truth.csv`), never `0`, `"-"`, or `"N/A"`.
  - In `FieldValue`, status must be one of `"ENCONTRADO"`, `"AUSENTE"`, or `"INVALIDO"`.
- **NIF Validation**: Portuguese NIFs must be 9 digits and pass modulo-11 check digit verification. Invalid NIFs are classified as `"INVALIDO"`.

## Module Contracts Already in Use

- **Preprocessing (`src/preprocessing.py`)**: `preprocess(image, PreprocessOptions()) -> PreprocessResult`. Five stages: resize (`max_side=1280`), orientation (aspect-ratio heuristic, then optional Tesseract OSD probe), perspective crop, morphological shadow division, CLAHE. `steps` keys are `"bruta"`, `"rodada"`, `"recortada"`, `"filtrada"` and feed the Gradio tab 2. `rotation_deg` is the cumulative clockwise rotation. `detect_orientation_osd` shells out to the Tesseract binary with a 5 s timeout and returns `0` on any failure, so it must never be a test dependency.
- **OCR (`src/ocr_engine.py`)**: `EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True, reader=None)`. The `reader` argument is the injection point for offline tests. Lines are sorted top-to-bottom then left-to-right and numbered from 1.
- **QR (`src/qr_reader.py`)**: `read(image, raw_image=None, detector=None) -> dict[str, str] | None`. Five decode stages (direct, grayscale, adaptive threshold, blob search with thermal dead-pin streak healing, raw-resolution retry). The parser keeps the raw AT keys (`A`, `B`, `F`, `H`, `O`, ...) and adds the aliases `nif_comerciante`, `nif_cliente`, `data`, `total`, `atcud`, `tipo_doc`. `B=999999990` means final consumer and maps to an empty `nif_cliente`. `pyzbar` is optional at runtime: if libzbar is missing the import fails silently and `cv2.QRCodeDetector` takes over.
- **Evidence (`src/evidence.py`)**: `draw_all_ocr_boxes` (all boxes plus `[id]` badges, tab 2), `highlight` (muted background plus semi-transparent highlight on cited lines, tab 1 and 3), `crop` (padded union box of cited lines, `None` when no id matches). All three are non-destructive and copy the input.
- **QR Code Reader usage**: Cross-validation and an extra interface column only; never the primary source, because payment terminal slips have no QR.
- **LLM Prompting Gotcha**: `llama-server` does not inject `response_format` schemas into the prompt text automatically. Prompts must explicitly describe the expected JSON schema. Keep schemas flat without union types to avoid breaking `llama.cpp`'s `json-schema-to-grammar`.

## Testing Rules

1. Tests must run completely offline: no network requests, no GPU requirements, no external API keys, and no heavy live model weights.
2. OCR tests must use synthetic images or mocked `OcrDocument` objects, never loading real EasyOCR weights in standard unit tests. Inject a fake reader (`EasyOCREngine(reader=...)`) or monkeypatch `src.ocr_engine.EasyOCREngine` in CLI tests.
3. Live model coverage belongs behind `@pytest.mark.slow` and must still pass with `uv run pytest -m slow` on a machine that has the weights cached.
4. Temporary artifacts must be written to pytest's `tmp_path` fixture, never committed to `data/` or root directories. Tests never read the real `data/receipts/` or `data/ground_truth.csv` files.
5. Tesseract-dependent tests monkeypatch the subprocess call; the suite must pass on a machine without the `tesseract` binary.

## Style & Workflow

- **No Emojis**: Do not use emojis anywhere (code, docstrings, commits, PRs, comments, or UI text).
- **Docstrings**: Google-style docstrings in English for all public functions and classes.
- **Print**: Only in the `main` CLI of a module, never in library code paths.
- **Git**: Trunk-based development with short-lived feature branches (`feat/wp-*`, `fix/*`). Target diff size < 200 lines per work package, and do not touch files outside the work package scope.
- **Docs**: `README.md` and `Docs/Planning/implementation_plan.md` describe what exists, not what is planned, except where a section is explicitly labelled as pending. Update both when a work package closes.
