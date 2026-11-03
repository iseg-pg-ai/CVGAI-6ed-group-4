# Repository Agent Guide (CVGAI-6ed-group-4)

Smart reader for Portuguese purchase receipts and simplified invoices. Hybrid pipeline: OpenCV preprocessing -> OCR (EasyOCR / Tesseract) -> structured extraction (Regex / Local LLM / Cloud LLM) -> verifiable Q&A with visual grounding -> Gradio UI.

## Current State

Closed work packages: WP-00 (skeleton), WP-01 (data contracts), WP-02 (pure normalizers), WP-03 (golden set loader), WP-04 (preprocessing), WP-05 (EasyOCR), WP-06 (Tesseract), WP-07 (AT QR reader), WP-08 (LLM client), WP-09 (regex baseline), WP-10 (LLM extractor), WP-11 (visual evidence), WP-12 (Q&A assistant), WP-13 (scoring), WP-14 (corpus statistics), WP-15 (benchmark runner), WP-16 (Gradio UI). `src/core/settings.py` was added outside the numbering as the single environment and path layer, and `scripts/check_golden_set.sh` provides automated golden set verification.

- Implemented: `src/core/types.py`, `src/core/settings.py`, `src/core/normalizers.py`, `src/core/cli_utils.py`, `src/evaluation/golden_set.py`, `src/vision/preprocessing.py`, `src/ocr/easyocr.py`, `src/ocr/tesseract.py`, `src/ocr/__init__.py`, `src/vision/qr_reader.py`, `src/extraction/llm_client.py`, `src/extraction/regex.py`, `src/extraction/llm.py`, `src/vision/evidence.py`, `src/extraction/qa_assistant.py`, `src/evaluation/scoring.py`, `src/evaluation/dataset_stats.py`, `src/evaluation/evaluate.py`, `src/app.py`, `scripts/run_llama_server.{sh,ps1,bat}`, `scripts/check_golden_set.sh`.
- Not implemented yet: `Docs/Presentation/` (WP-18 slide deck). Do not describe it as available, in docs or in code.
- Corpus: `data/receipts/` holds 27 images (`01_...` to `27_...`); `data/ground_truth.csv` has 27 records validated structurally by `scripts/check_golden_set.sh`, with WP-17 in progress for semantic alignment of the `entidade` column against printed receipt headers.
- Suite: 291 tests, offline, about 4.5 s. One `@pytest.mark.slow` live-EasyOCR test is deselected by default.

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

- **Keystone (`src/core/types.py`)**: All pipeline stages exchange only classes from `src/core/types.py` (`BBox`, `OcrLine`, `OcrDocument`, `OcrEngine`, `FieldValue`, `Extraction`, `GroundTruthRecord`, `MatchResult`, `Answer`). Also defines and exports shared constants (`FIELD_NAMES`, `GROUPS`, `OCR_ENGINES`, `LLM_PROVIDERS`, `EXTRACTION_METHODS`, `CONSUMIDOR_FINAL_NIF`, `EARLIEST_PLAUSIBLE_DATE`, `IMAGE_EXTENSIONS`, `STATUSES`, `ANSWER_STATUSES`, `GROUND_TRUTH_COLUMNS`, `MATCH_RESULT_COLUMNS`, `METRIC_NAMES`, `ROTATION_CHOICES`). Strict `Literal` aliases (`OcrEngineName`, `MethodName`, `GroupName`, `FieldName`, `ProviderName`, `Status`, `AnswerStatus`, `MetricName`, `RotationOption`) are enforced on dataclass fields. Do not modify contracts in `types.py` without coordinating changes.
- **Bounding boxes**: `BBox(x1, y1, x2, y2)` is in pixels, in the reference frame of the preprocessed image, with `x2`/`y2` exclusive. `BBox.__post_init__` enforces geometric non-negative ordering (`x2 >= x1` and `y2 >= y1`). Every OCR engine must clamp coordinates to the image.
- **Settings (`src/core/settings.py`)**: The only module allowed to read environment variables or resolve repository paths. `load_settings(repo_root=...)` loads values without mutating `os.environ` (via `dotenv_values`), logs warnings on unrecognized `OCR_ENGINE` values, and exposes `max_side`, `clahe_clip_limit`, `llama_port`, and `llama_ngl`. `get_settings()` is `lru_cache`d, so tests that change the environment must use `load_settings`, never the cached getter.
- **CLI convention**: Every runnable module exposes `main(argv: Sequence[str] | None = None, settings_override: Settings | None = None)` plus the `if __name__ == "__main__": main()` guard. Heavy imports (Pillow, EasyOCR, `src.core.settings`) are deferred to `main` or to the engine constructor, so importing a module stays cheap for tests.
- **Image validation**: Every public function that takes an image raises `ValueError("Input image must not be empty or zero-sized.")` on empty input and returns a new array instead of mutating the caller's one.
- **Canonical Fields**: The 5 required extraction fields are `entidade`, `nif_comerciante`, `nif_cliente`, `data`, and `total`.
- **Absent vs Invalid**:
  - In `GroundTruthRecord`, absent fields are `None` (empty CSV field in `data/ground_truth.csv`), never `0`, `"-"`, or `"N/A"`.
  - In `FieldValue`, status must be one of `"ENCONTRADO"`, `"AUSENTE"`, or `"INVALIDO"`.
- **NIF Validation**: Portuguese NIFs must be 9 digits and pass modulo-11 check digit verification. Invalid NIFs are classified as `"INVALIDO"`.

## Module Contracts Already in Use

- **Preprocessing (`src/vision/preprocessing.py`)**: `preprocess(image, PreprocessOptions()) -> PreprocessResult`. Five stages: resize (`max_side=1280`), orientation (aspect-ratio heuristic), 4-point adaptive perspective crop (with `minAreaRect` fallback and optional text-line deskew), morphological shadow division, CLAHE contrast enhancement, and upside-down flip verification (OpenCV QR vertical centroid heuristic `detect_upside_down_qr`, followed by Tesseract OSD probe `detect_orientation_osd`). `steps` keys are `"bruta"`, `"rodada"`, `"recortada"`, `"filtrada"` and feed Gradio tab 2. `tesseract_cmd` is forwarded through `PreprocessOptions`.
- **OCR (`src/ocr/easyocr.py`)**: `EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True, reader=None)`. The `reader` argument is the injection point for offline tests. Multi-column fragments on the same baseline (>45% vertical overlap or centroid distance <=45% of min height) are merged left-to-right into unified `OcrLine` entries, sorted top-to-bottom and numbered from 1.
- **QR (`src/vision/qr_reader.py`)**: `read(image, raw_image=None, detector=None) -> dict[str, str] | None`. Five decode stages (direct, grayscale, adaptive threshold, blob search with thermal dead-pin streak healing and area bounds, raw-resolution retry). `_decode_all_codes` loops across all detected barcodes (`pyzbar.decode` and `detectAndDecodeMulti`) so promotional QR codes never mask the fiscal AT QR code. Parser keeps raw AT keys and adds canonical aliases (`nif_comerciante`, `nif_cliente`, `data`, `total`, `atcud`, `tipo_doc`). `B=999999990` maps to empty `nif_cliente`.
- **Evidence (`src/vision/evidence.py`)**: `draw_all_ocr_boxes` (all boxes plus `[id]` badges, tab 2), `highlight` (muted background plus semi-transparent highlight on cited lines, tab 1 and 3), `crop` (padded union box of cited lines, `None` when no id matches). All three are non-destructive and copy the input.
- **Normalizers (`src/core/normalizers.py`)**: pure functions, no I/O, no environment. `limpar_texto`, `parse_moeda`, `parse_data` (iterates all candidate matches via `re.finditer`), `parse_nif`, `validar_nif`, `normalizar_entidade` (strips `{"lda", "sa", "unipessoal", "sociedade"}` without stripping single-word names), `valores_equivalentes(campo, esperado, obtido)`.
- **Tesseract (`src/ocr/tesseract.py`)**: `TesseractEngine(lang="por", psm=6, tesseract_cmd=None, image_to_data_fn=None)`, aggregates word boxes by `(block_num, par_num, line_num)`, discards `conf < 0`, and restores global `tesseract_cmd`. `image_to_data_fn` is the test injection hook.
- **Golden Set (`src/evaluation/golden_set.py`)**: `carregar_ground_truth(path, receipts_dir=None) -> list[GroundTruthRecord]` and `validar_ground_truth(path, receipts_dir) -> list[str]`. Validates header, existence of images, groups in `{"direito", "amarrotado"}`, valid NIFs (rejecting `999999990` with guidance), dates (not in future, not before `2000-01-01`), and positive finite totals. Raises `GoldenSetError`.
- **LLM client (`src/extraction/llm_client.py`)**: `build_client(provider, settings=None, primary_client=None, fallback_client=None) -> LLMClient` with `provider` in `llm_local`, `llm_gemini`. `complete(system, user, schema, image=None)` returns raw text; failover records the provider that actually answered in `last_effective_provider`. `GeminiClient` strips unsupported `additionalProperties` recursively via `_sanitize_gemini_schema`.
- **Regex baseline (`src/extraction/regex.py`)**: `extract(doc, qr=None) -> Extraction` with `method="regex"`. `_TOTAL_LINE` supports optional qualifiers (`A PAGAR`, `LIQUIDO`, `GERAL`, `FINAL`) and currency prefixes (`EUR`, `€`). `CONSUMIDOR_FINAL_NIF` (`999999990`) is mapped to `AUSENTE`. Client NIF duplicate of merchant is rejected.
- **LLM extractor (`src/extraction/llm.py`)**: `extract(doc, qr=None, client=None) -> Extraction`. Re-parses NIFs, dates, and totals with pure normalisers. Grounding `_locate_line` for `total` scans lines in reverse (`reversed(lines)`) prioritizing lines with total keywords. Strict schema with `additionalProperties: false` is enforced on `llama-server`.
- **Q&A assistant (`src/extraction/qa_assistant.py`)**: `ask(question, doc, client=None, provider="llm_local") -> Answer`. Returns `status` in `{"ENCONTRADO", "AUSENTE NO DOCUMENTO"}` and re-anchors missing `line_ids` using punctuation-stripped, stopword-filtered tokens.
- **Scoring (`src/evaluation/scoring.py`)**: `pontuar_registo(verdade, extracao) -> list[MatchResult]`, `agregar(resultados) -> AggregatedMetrics`, `escrever_tabelas(resultados, destino) -> None`. Fields marked `"INVALIDO"` or `"AUSENTE"` never score `acertou=True` when `esperado is not None`.
- **Corpus statistics (`src/evaluation/dataset_stats.py`)**: `resumo(registos) -> dict[str, object]` and `imprimir_resumo(registos) -> None`.
- **Benchmark runner (`src/evaluation/evaluate.py`)**: `avaliar_registo(...) -> tuple[Extraction, list[MatchResult]]`, `avaliar_corpus(...) -> list[MatchResult]`. Preprocesses, runs OCR, extracts canonical fields, scores each receipt against reference truth via `pontuar_registo`, and optionally outputs `metrics.csv` and `metrics.md` with run configuration metadata via `escrever_tabelas`.
- **Gradio UI (`src/app.py`)**: `build_app(settings_override=None) -> gr.Blocks`. Three tabs: Document Reader and Q&A (with decoupled Q&A provider selector), Step-by-Step CV Pipeline, and Validation & Benchmark Dashboard (reading `metrics.csv` with `utf-8-sig`). In-memory caches are guarded by `_CACHE_LOCK`.
- **QR Code Reader usage**: Cross-validation and an extra interface column only; never the primary source, because payment terminal slips have no QR.
- **LLM Prompting Gotcha**: `llama-server` does not inject `response_format` schemas into the prompt text automatically. Prompts must explicitly describe the expected JSON schema. Keep schemas flat without union types to avoid breaking `llama.cpp`'s `json-schema-to-grammar`.
- **Local Model Serving (`scripts/run_llama_server.*`)**: Defaults to `models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` on port 8080 with `NGL=0`. `NGL=0` runs safely on CPU; passing `NGL=99` on small GPUs (e.g. 2 GB MX150) fails allocation instead of falling back. `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` can also serve as text, but image inputs fail with HTTP 500 because llama.cpp requires a companion `mmproj` GGUF file for vision.
- **Golden Set Check Script (`scripts/check_golden_set.sh`)**: Convenience POSIX shortcut that runs `python -m src.evaluation.golden_set` and `python -m src.evaluation.dataset_stats`.

## Testing Rules

1. Tests must run completely offline: no network requests, no GPU requirements, no external API keys, and no heavy live model weights.
2. OCR tests must use synthetic images or mocked `OcrDocument` objects, never loading real EasyOCR weights in standard unit tests. Inject a fake reader (`EasyOCREngine(reader=...)`) or monkeypatch `src.ocr.easyocr.EasyOCREngine` in CLI tests.
3. Live model coverage belongs behind `@pytest.mark.slow` and must still pass with `uv run pytest -m slow` on a machine that has the weights cached.
4. Temporary artifacts must be written to pytest's `tmp_path` fixture, never committed to `data/` or root directories. Tests never read the real `data/receipts/` or `data/ground_truth.csv` files.
5. Tesseract-dependent tests monkeypatch the subprocess call; the suite must pass on a machine without the `tesseract` binary.

## Style & Workflow

- **No Emojis**: Do not use emojis anywhere (code, docstrings, commits, PRs, comments, or UI text).
- **Docstrings**: Google-style docstrings in English for all public functions and classes.
- **Print**: Only in the `main` CLI of a module, never in library code paths.
- **Git**: Trunk-based development with short-lived feature branches (`feat/wp-*`, `fix/*`). Target diff size < 200 lines per work package, and do not touch files outside the work package scope.
- **Docs**: `README.md` and `Docs/Planning/implementation_plan.md` describe what exists, not what is planned, except where a section is explicitly labelled as pending. Update both when a work package closes.
