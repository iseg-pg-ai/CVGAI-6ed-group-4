# Repository Agent Guide (CVGAI-6ed-group-4)

Smart reader for Portuguese purchase receipts and simplified invoices. Hybrid pipeline: OpenCV preprocessing -> OCR (EasyOCR / Tesseract) -> structured extraction (Regex / Local LLM / Cloud LLM) -> verifiable Q&A with visual grounding -> Gradio UI.

## Current State

Closed work packages: WP-00 (skeleton), WP-01 (data contracts), WP-02 (pure normalizers), WP-03 (golden set loader), WP-04 (preprocessing), WP-05 (EasyOCR), WP-06 (Tesseract), WP-07 (AT QR reader), WP-08 (LLM client), WP-09 (regex baseline), WP-10 (LLM extractor), WP-11 (visual evidence), WP-12 (Q&A assistant), WP-13 (scoring), WP-14 (corpus statistics), WP-15 (benchmark runner), WP-16 (Gradio UI), WP-17 (golden set `entidade` alignment). `src/core/settings.py` uses `pydantic-settings` to centralize configuration, `src/core/logging.py` centralizes logging, and `./run` / `run.bat` (`poethepoet`) provide unified cross-platform pipeline tasks (`check`, `check-dataset`, `evaluate`, `app`, `serve`, `hooks`).

- Implemented: `src/core/types.py`, `src/core/settings.py`, `src/core/logging.py`, `src/core/server_launcher.py`, `src/core/normalizers.py`, `src/core/cli_utils.py`, `src/evaluation/golden_set.py`, `src/evaluation/scoring.py`, `src/evaluation/dataset_stats.py`, `src/evaluation/tune_preprocessing.py`, `src/evaluation/evaluate.py`, `src/vision/preprocessing.py`, `src/vision/geometry.py`, `src/vision/photometry.py`, `src/vision/orientation.py`, `src/vision/qr_reader.py`, `src/vision/evidence.py`, `src/ocr/easyocr.py`, `src/ocr/tesseract.py`, `src/ocr/__init__.py`, `src/extraction/llm_client.py`, `src/extraction/llm_client_base.py`, `src/extraction/llm_providers.py`, `src/extraction/llm_prompts.py`, `src/extraction/llm_grounding.py`, `src/extraction/regex.py`, `src/extraction/regex_nifs.py`, `src/extraction/llm.py`, `src/extraction/qa_assistant.py`, `src/app.py`, `src/app_layout.py`, `run`, `run.bat`.
- Not implemented yet: `Docs/Presentation/` (WP-18 slide deck). Do not describe it as available, in docs or in code.
- Corpus: `data/receipts/` holds 27 images (`01_...` to `27_...`); `data/ground_truth.csv` has 27 records validated by `./run check-dataset`. Note that crumpled receipt filenames on disk use the `NN_amachucado_...` prefix, whereas the CSV `grupo` column and `GroupName` contract strictly use `"amarrotado"`.
- Suite: 320 offline tests (~4.5 s) across 19 test files. One `@pytest.mark.slow` live-EasyOCR test is deselected by default.

Each module docstring carries a `Ref:` line pointing at the section of `Docs/implementation_plan.md` that specifies it. Keep that convention for new modules.

## Environment & Toolchain

- **Manager**: `uv` only (do not use raw `pip`, `poetry`, or `conda`).
- **Python Version**: Strictly `>=3.12,<3.13` (enforced in `pyproject.toml` and `.python-version`). EasyOCR drags `torch`/`torchvision` which lacks stable wheels on >=3.14.
- **OpenCV Constraint**: Use only `opencv-python-headless`. Never install `opencv-contrib-python` or `opencv-python`; having multiple distributions claiming `cv2` corrupts the environment. `cv2.QRCodeDetector` is fully functional in headless (verified on OpenCV 5).
- **Execution Mode**: `[tool.uv] package = false`. Run modules as `uv run python -m src.<subpackage>.<module>` (or `uv run python -m src.app`) from the repository root; never install `src` as an editable distribution.
- **Windows Gotcha**: Set `$env:PYTHONUTF8 = "1"` before running CLI scripts to prevent `UnicodeEncodeError` on Portuguese characters. Always use `pathlib.Path`, never hardcoded backslashes or `os.sep`.

## Verification Commands

Run checks before completing work (or simply run `./run check` / `run check` on Windows):

```bash
# Unified 4-step verification gate
./run check

# Or individual commands in order:
# 1. Formatting (line length 120; also set in pyproject.toml)
uv run ruff format --line-length 120 src tests

# 2. Linting
uv run ruff check src tests

# 3. Type Checking
uv run pyright src tests

# 4. Test Suite (offline and fast)
uv run pytest
```

### Focused Testing & Dataset Shortcuts

```bash
./run check-dataset                                 # Validate ground_truth.csv + print corpus stats
uv run pytest tests/test_evidence.py -q             # Single test file
uv run pytest -k "nif" -q                           # Filter tests by expression
uv run pytest -m slow                               # Opt-in live EasyOCR test (needs cached weights)
uv run pytest --cov=src --cov-report=term-missing   # Coverage report
```

`pyproject.toml` sets `addopts = "-q --strict-markers -m 'not slow'"`, so `slow` tests never run by default and an unregistered marker fails the suite.

## Architecture & Data Contracts

- **Keystone (`src/core/types.py`)**: All pipeline stages exchange only classes from `src/core/types.py` (`BBox`, `OcrLine`, `OcrDocument`, `OcrEngine`, `FieldValue`, `Extraction`, `GroundTruthRecord`, `MatchResult`, `Answer`). Also exports shared constants (`FIELD_NAMES`, `GROUPS`, `OCR_ENGINES`, `LLM_PROVIDERS`, `EXTRACTION_METHODS`, `CONSUMIDOR_FINAL_NIF`, `EARLIEST_PLAUSIBLE_DATE`, `IMAGE_EXTENSIONS`, `STATUSES`, `ANSWER_STATUSES`, `GROUND_TRUTH_COLUMNS`, `MATCH_RESULT_COLUMNS`, `METRIC_NAMES`, `ROTATION_CHOICES`), strict `Literal` aliases (`OcrEngineName`, `MethodName`, `GroupName`, `FieldName`, `ProviderName`, `Status`, `AnswerStatus`, `MetricName`, `RotationOption`), and `validate_image(image)`. Do not modify contracts in `types.py` without coordinating changes.
- **Bounding boxes**: `BBox(x1, y1, x2, y2)` is in pixels, in the reference frame of the preprocessed image, with `x2`/`y2` exclusive. `BBox.__post_init__` enforces geometric non-negative ordering (`x2 >= x1` and `y2 >= y1`). Every OCR engine must clamp coordinates to the image.
- **Settings (`src/core/settings.py`)**: Centralizes environment variables and repository paths using `pydantic-settings` (`BaseSettings` with `frozen=True`, `extra="ignore"`). Dynamically resolves relative paths against `repo_root`, supports legacy environment variable aliases (`PORT`, `NGL`, `CTX`, `MODEL`, `PREPROCESS_*`), and exposes `__dataclass_fields__` so `dataclasses.replace(settings, ...)` remains compatible. Settings attributes include `max_side`, `clahe_clip_limit`, `contrast_gamma`, `sharpen_strength`, `shadow_kernel_size`, `llama_port`, `llama_ngl`, `llama_ctx`, and `llama_model_path`. `src/core/logging.py` configures unified console logging via `setup_logging()` reading `LOG_LEVEL` (default `INFO`). `get_settings()` is `lru_cache`d, so tests that isolate paths or env vars must use `load_settings(repo_root=...)`, never the cached getter.
- **CLI convention**: Every runnable module exposes `main(argv: Sequence[str] | None = None, settings_override: Settings | None = None)` plus the `if __name__ == "__main__": main()` guard, using shared helpers in `src/core/cli_utils.py` (`list_receipt_images`, `resolve_and_load_receipt`, `format_extraction_field`). Heavy imports (Pillow, EasyOCR, `src.core.settings`) are deferred to `main` or to the engine constructor so importing a module stays cheap for tests.
- **Image validation**: Every public function that takes an image calls `validate_image(image)` (raising `ValueError("Input image must not be empty or zero-sized.")` on empty input) and returns a new array instead of mutating the caller's one.
- **Canonical Fields**: The 5 required extraction fields are `entidade`, `nif_comerciante`, `nif_cliente`, `data`, and `total`.
- **Absent vs Invalid**:
  - In `GroundTruthRecord`, absent fields are `None` (empty CSV field in `data/ground_truth.csv`), never `0`, `"-"`, or `"N/A"`.
  - In `FieldValue`, status must be one of `"ENCONTRADO"`, `"AUSENTE"`, or `"INVALIDO"`.
- **NIF Validation**: Portuguese NIFs must be 9 digits with a valid first digit (`12356789`) and pass modulo-11 check digit verification. Invalid NIFs are classified as `"INVALIDO"`. `CONSUMIDOR_FINAL_NIF` (`999999990`) passes modulo-11 but is mapped to `"AUSENTE"` by extractors and rejected by `golden_set.py`.

## Module Contracts & Gotchas

- **Preprocessing (`src/vision/preprocessing.py`)**: Facade over `src/vision/geometry.py`, `src/vision/photometry.py`, and `src/vision/orientation.py`. `preprocess(image, PreprocessOptions()) -> PreprocessResult`. Stages: resize (`max_side=1280`), orientation (aspect-ratio heuristic), 4-point adaptive perspective crop (with `minAreaRect` fallback and optional text-line deskew), morphological shadow attenuation (kernel 25), CLAHE contrast enhancement (`clip_limit=2.0`), gamma ink density correction, unsharp mask sharpening (`strength=0.35` by default), and upside-down flip verification (`resolve_semantic_flip` combining OpenCV QR vertical centroid heuristic `detect_qr_vertical_ratio` and Tesseract OSD probe `detect_orientation_osd`). `steps` keys are `"bruta"`, `"rodada"`, `"recortada"`, `"filtrada"` and feed Gradio tab 2. `tesseract_cmd` is forwarded through `PreprocessOptions`.
- **OCR (`src/ocr/easyocr.py`, `src/ocr/tesseract.py`, `src/ocr/__init__.py`)**:
  - `build_ocr_engine(engine_name, settings, gpu=False, warmup=False)` instantiates the active engine.
  - `EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True, reader=None)`: `reader` is the test injection hook. `read` passes `width_ths=0.5, mag_ratio=1.5` to `readtext` (falling back on `TypeError` for simple mocks). Multi-column fragments on the same baseline (>45% vertical overlap or centroid distance <=45% of min height) are merged left-to-right into unified `OcrLine` entries, sorted top-to-bottom and numbered from 1.
  - `TesseractEngine(lang="por", psm=6, tesseract_cmd=None, image_to_data_fn=None)`: aggregates word boxes by `(block_num, par_num, line_num)`, discards `conf < 0`, and restores global `tesseract_cmd`. `image_to_data_fn` is the test injection hook.
- **QR (`src/vision/qr_reader.py`)**: `read(image, raw_image=None, detector=None) -> dict[str, str] | None`. Cross-validation and extra UI column only (never the primary extraction source). Five decode stages (direct, grayscale, adaptive threshold, blob search with thermal dead-pin streak healing and proportional area bounds, raw-resolution retry). `_decode_all_codes` loops across all detected barcodes (`pyzbar.decode` and `detectAndDecodeMulti`) so promotional QR codes never mask the fiscal AT QR code. `B=999999990` maps to empty `nif_cliente`. Also exports `find_qr_bounding_boxes` and `mask_qr_region(image, padding=12)` used before OCR in `app.py` and `evaluate.py` to prevent printed QR codes from inducing hallucinated OCR text lines.
- **Evidence (`src/vision/evidence.py`)**: `draw_all_ocr_boxes` (all boxes plus `[id]` badges, tab 2), `highlight` (muted background plus semi-transparent highlight on cited lines, tab 1 and 3), `crop` (padded union box of cited lines, `None` when no id matches). All three are non-destructive and copy the input.
- **Normalizers (`src/core/normalizers.py`)**: Pure functions, no I/O, no environment. `limpar_texto`, `parse_moeda`, `parse_data` (iterates all candidate matches via `re.finditer`), `validar_data` (`2000-01-01 <= parsed <= today`), `parse_nif` (handles OCR-garbled `PT` prefixes, 10-digit leading-`7` artifacts, and `/`->`7`/`1` repairs validated by check digit), `validar_nif`, `normalizar_entidade` (strips `ENTITY_SUFFIXES = {"lda", "sa", "unipessoal", "unip", "sociedade", "limitada", "s"}` without stripping single-word names), and `valores_equivalentes(campo, esperado, obtido)` (validates `campo` before `None` checks; `entidade` accepts token-boundary containment with len >= 4 or `SequenceMatcher` ratio >= `ENTITY_SIMILARITY_THRESHOLD` at 0.72 across full strings or sliding token windows).
- **Golden Set (`src/evaluation/golden_set.py`)**: `carregar_ground_truth(path, receipts_dir=None) -> list[GroundTruthRecord]` and `validar_ground_truth(path, receipts_dir) -> list[str]`. Validates header, existence of images, groups in `{"direito", "amarrotado"}`, valid NIFs (rejecting `999999990` with guidance), dates (`2000-01-01` to today), and positive finite totals. Raises `GoldenSetError`.
- **LLM client (`src/extraction/llm_client.py`, `src/extraction/llm_client_base.py`, `src/extraction/llm_providers.py`)**: `build_client(provider, settings=None, primary_client=None, fallback_client=None) -> LLMClient` with `provider` in `llm_local`, `llm_gemini`. `complete(system, user, schema=None)` returns raw text; failover records the answering backend in `last_effective_provider`. `OpenAICompatibleClient` serves local `llama-server` via the OpenAI SDK; `GeminiClient` connects via `google-genai` and strips `additionalProperties` recursively via `sanitize_gemini_schema`. Also exports shared helpers `build_object_schema`, `parse_json_object`, and `coerce_line_id`.
- **Regex baseline (`src/extraction/regex.py`, `src/extraction/regex_nifs.py`)**: `extract(doc, qr=None) -> Extraction` with `method="regex"`. `extract_entity` bounds search to the top 40% header band (`HEADER_BAND_RATIO = 0.4`). `_TOTAL_LINE` supports optional qualifiers (`A PAGAR`, `LIQUIDO`, `GERAL`, `FINAL`) and currency prefixes (`EUR`, `€`). `extract_nifs` filters out customer support markers and toll-free telephone prefixes (`800`, `808`, `707`, `708`), and rejects client NIFs duplicating the merchant NIF.
- **LLM extractor (`src/extraction/llm.py`, `src/extraction/llm_prompts.py`, `src/extraction/llm_grounding.py`)**: `extract(doc, qr=None, client=None) -> Extraction`. Re-parses NIFs, dates, and totals with pure normalizers; `build_field` strips trailing tax markers (`NIF`, `NIPC`, `Contribuinte`) from `entidade`; `_reconcile_nifs` reconciles `nif_comerciante` and `nif_cliente` against regex OCR extraction, dropping helpline/toll-free numbers (`800`, `808`, `707`, `708`); falls back to regex `_extract_date` and `_extract_total` when the LLM returns `AUSENTE` or when the LLM returns an implausible total `> 500.00` without decimals; grounds `total` lines in reverse (`reversed(lines)`) prioritizing lines with total keywords.
- **Q&A assistant (`src/extraction/qa_assistant.py`)**: `ask(question, doc, client=None, provider="llm_local") -> Answer`. Returns `status` in `{"ENCONTRADO", "AUSENTE NO DOCUMENTO"}` and re-anchors missing `line_ids` using punctuation-stripped, stopword-filtered tokens.
- **Scoring & Stats (`src/evaluation/scoring.py`, `src/evaluation/dataset_stats.py`, `src/evaluation/tune_preprocessing.py`)**: `pontuar_registo(verdade, extracao) -> list[MatchResult]`, `agregar(resultados) -> AggregatedMetrics`, `escrever_tabelas(resultados, destino, metadata=None) -> None` (writes `metrics.csv` with `utf-8-sig` and `metrics.md`). Fields marked `"INVALIDO"` or `"AUSENTE"` never score `acertou=True` when `esperado is not None`. `tune_preprocessing.py` performs coordinate-ascent hyperparameter tuning on photometric options.
- **Benchmark runner (`src/evaluation/evaluate.py`)**: `avaliar_registo(...)` and `avaliar_corpus(...)`. Includes a post-OCR fiscal-keyword orientation check that flips the preprocessed image 180 degrees and re-runs OCR if the initial pass finds `<= 2` receipt keywords and the flipped pass yields `>= 3` keywords (and at least +3 over the initial pass). Masks QR regions before OCR inference.
- **Gradio UI (`src/app.py`, `src/app_layout.py`)**: `build_app(settings_override=None) -> gr.Blocks`. Three tabs: Document Reader and Q&A (with decoupled Q&A provider selector and `gr.Warning` toast fallback to regex), Step-by-Step CV Pipeline (inspecting 4 OpenCV stages plus OCR bounding boxes), and Validation & Benchmark Dashboard (reading `metrics.csv` with `utf-8-sig`). In-memory caches are guarded by `_CACHE_LOCK`.
- **LLM Prompting Gotcha**: `llama-server` does not inject `response_format` schemas into the prompt text automatically. Prompts must explicitly describe the expected JSON schema. Keep schemas flat without union types to avoid breaking `llama.cpp`'s `json-schema-to-grammar`.
- **Local Model Serving (`./run serve` / `run serve` via `src/core/server_launcher.py`)**: Defaults to `models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` on port 8080 with `NGL=0` and `CTX=8192`. `NGL=0` runs safely on CPU; passing `-ngl 99` offloads layers to GPU. Accepts a positional model path argument (e.g. `./run serve models/Qwen2.5-7B-Instruct-Q8_0.gguf`), `--port`, `-ngl`, `-c`, and `--dry-run`.

## Testing Rules

1. Tests must run completely offline: no network requests, no GPU requirements, no external API keys, and no heavy live model weights.
2. OCR tests must use synthetic images or mocked `OcrDocument` objects, never loading real EasyOCR weights in standard unit tests. Inject a fake reader (`EasyOCREngine(reader=...)`) or monkeypatch `src.ocr.easyocr.EasyOCREngine` in CLI tests.
3. Live model coverage belongs behind `@pytest.mark.slow` and must still pass with `uv run pytest -m slow` on a machine that has the weights cached.
4. Temporary artifacts must be written to pytest's `tmp_path` fixture, never committed to `data/` or root directories. Tests never read the real `data/receipts/` or `data/ground_truth.csv` files.
5. Tesseract-dependent tests monkeypatch the subprocess call or pass `image_to_data_fn`; the suite must pass on a machine without the `tesseract` binary.

## Style & Workflow

- **No Emojis**: Do not use emojis anywhere (code, docstrings, commits, PRs, comments, or UI text).
- **Docstrings**: Google-style docstrings in English for all public functions and classes.
- **Print**: Only in the `main` CLI of a module (and `src/core/cli_utils.py`), never in library code paths.
- **Git**: Trunk-based development with short-lived feature branches (`feat/wp-*`, `fix/*`). Target diff size < 200 lines per work package, and do not touch files outside the work package scope.
- **Docs**: `README.md` and `Docs/implementation_plan.md` describe what exists, not what is planned, except where a section is explicitly labelled as pending. Update both when a work package closes.
