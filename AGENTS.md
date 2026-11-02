# Repository Agent Guide (CVGAI-6ed-group-4)

Smart reader for Portuguese purchase receipts and simplified invoices. Hybrid pipeline: OpenCV preprocessing -> OCR (EasyOCR / Tesseract) -> structured extraction (Regex / Local LLM / Cloud LLM) -> verifiable Q&A with visual grounding -> Gradio UI.

## Current State

Closed work packages: WP-00 (skeleton), WP-01 (data contracts), WP-02 (pure normalizers), WP-03 (golden set loader), WP-04 (preprocessing), WP-05 (EasyOCR), WP-06 (Tesseract), WP-07 (AT QR reader), WP-08 (LLM client), WP-09 (regex baseline), WP-10 (LLM extractor), WP-11 (visual evidence), WP-12 (Q&A assistant), WP-13 (scoring), WP-14 (corpus statistics), WP-16 (Gradio UI). `src/settings.py` was added outside the numbering as the single environment and path layer.

- Implemented: `src/types.py`, `src/settings.py`, `src/normalizers.py`, `src/golden_set.py`, `src/preprocessing.py`, `src/ocr_engine.py`, `src/ocr_tesseract.py`, `src/qr_reader.py`, `src/llm_client.py`, `src/extractor_regex.py`, `src/extractor_llm.py`, `src/evidence.py`, `src/qa_assistant.py`, `src/scoring.py`, `src/dataset_stats.py`, `src/app.py`, `scripts/run_llama_server.{sh,ps1,bat}`.
- Not implemented yet: `evaluate.py`, `scripts/check_golden_set.sh`. Do not describe them as available, in docs or in code.
- Corpus: `data/receipts/` holds 9 images; `data/ground_truth.csv` has only the header (WP-17 is open).
- Suite: 263 tests, offline, about 4 s. One `@pytest.mark.slow` live-EasyOCR test is deselected by default.

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
- **Normalizers (`src/normalizers.py`)**: pure functions, no I/O, no environment. `limpar_texto`, `parse_moeda` (returns the **first** number in the string), `parse_data` (hyphen dates are read as `YY-MM-DD`), `parse_nif`, `validar_nif`, `normalizar_entidade`, `valores_equivalentes(campo, esperado, obtido)`.
- **Tesseract (`src/ocr_tesseract.py`)**: `TesseractEngine(lang="por", psm=6)`, aggregates word boxes by `(block_num, par_num, line_num)` and discards `conf < 0`.
- **LLM client (`src/llm_client.py`)**: `build_client(provider, settings=None, primary_client=None, fallback_client=None) -> LLMClient` with `provider` in `llm_local`, `llm_gemini`, `llm_openai`. `complete(system, user, schema)` returns raw text; failover records the provider that actually answered in `last_effective_provider`. Clients are injectable, so tests never touch the network.
- **Regex baseline (`src/extractor_regex.py`)**: `extract(doc, qr=None) -> Extraction` with `method="regex"`. The QR payload is accepted for interface parity and never fills a field. `999999990` (final consumer) is rejected by the explicit `CONSUMIDOR_FINAL_NIF` constant, because it passes the NIF check digit. An `INVALIDO` NIF keeps its digits so scoring counts it as wrong rather than as an omission.
- **LLM extractor (`src/extractor_llm.py`)**: `extract(doc, qr=None, client=None) -> Extraction`; without a client it builds the configured provider from the environment, so tests always inject a double. Returns `method=client.last_effective_provider` and keeps `raw_response`. Raises `LLMResponseError` on malformed output instead of returning five `AUSENTE`. The model is never trusted on machine-checkable parts: NIFs, dates and totals are re-parsed with the pure normalisers, the `presente` flag overrides any contradictory text, and any `line_id` that is missing, zero, or does not exist is re-grounded deterministically by `_locate_line` in the OCR text (the live 3B model returned `0` for four fields on a long receipt while extracting values correctly). The prompt explicitly forbids calculating totals; without that, the live model summed article lines into a false-positive total on a receipt with no printed total. `SCHEMA` sets `additionalProperties: false` on every object, which OpenAI strict mode requires and Gemini accepts.
- **QR Code Reader usage**: Cross-validation and an extra interface column only; never the primary source, because payment terminal slips have no QR.
- **LLM Prompting Gotcha**: `llama-server` does not inject `response_format` schemas into the prompt text automatically. Prompts must explicitly describe the expected JSON schema. Keep schemas flat without union types to avoid breaking `llama.cpp`'s `json-schema-to-grammar`.
- **Local Model Serving (`scripts/run_llama_server.*`)**: Defaults to `models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` on port 8080 with `NGL=0`. `NGL=0` runs safely on CPU; passing `NGL=99` on small GPUs (e.g. 2 GB MX150) fails allocation instead of falling back. `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` can also serve as text, but image inputs fail with HTTP 500 because llama.cpp requires a companion `mmproj` GGUF file for vision.

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
