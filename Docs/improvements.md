# Project Audit and Technical Improvements Specification

Repository: `CVGAI-6ed-group-4`  
Reference Date: November 2026  
Domain: Smart Reader for Portuguese Receipts and Simplified Invoices (CVGAI 6th Edition)

---

## 1. Executive Summary and Status Overview

This document provides an exhaustive, module-by-module technical audit of the codebase, identifying functional gaps, edge-case vulnerabilities, architectural inconsistencies, and necessary improvements across all pipeline stages.

### Status Summary
- **Implemented Modules & Scripts**: `src/types.py`, `src/settings.py`, `src/normalizers.py`, `src/golden_set.py`, `src/preprocessing.py`, `src/ocr_engine.py`, `src/ocr_tesseract.py`, `src/qr_reader.py`, `src/llm_client.py`, `src/extractor_regex.py`, `src/extractor_llm.py`, `src/evidence.py`, `src/qa_assistant.py`, `src/scoring.py`, `src/dataset_stats.py`, `src/app.py`, `scripts/run_llama_server.{sh,ps1,bat}`, `scripts/check_golden_set.sh`.
- **Addressed Technical Audits**: The module-level robustness fixes, typing enforcements, and script alignments identified in Section 4 were implemented across commits `66209907`, `3d4c7805`, and `8ae84b28`.
- **Remaining Work Packages**:
  - `src/evaluate.py` & `tests/test_evaluate.py` (WP-15 - Automated Benchmark Runner)
  - `data/ground_truth.csv` annotations audit (WP-17 - semantic alignment of `entidade` trading names)
  - `Docs/Presentation/` (WP-18 - Golden Set and Project Slide Deck)

---

## 2. Unfinished Work Packages and Core Functional Gaps

### 2.1 WP-15: Automated Benchmark Runner (`src/evaluate.py` & `tests/test_evaluate.py`)
- **Current State**: Pending implementation.
- **Consequence**: Users cannot execute the end-to-end evaluation benchmark over `data/ground_truth.csv` to output `results/metrics.csv` and `results/metrics.md`. Consequently, Tab 3 of the Gradio interface (`src/app.py`) displays only a placeholder notification.
- **Architectural Requirements & Deficiencies in Existing Plan**:
  - The specification in `Docs/Planning/implementation_plan.md` (lines 776-876) describes evaluating a single `--engine` and single `--method` per execution, overwriting `results/metrics.csv`.
  - In contrast, `PROJECT_PLAN.md` (Sections 4 and 5) mandates a comparative assessment across OCR engines (EasyOCR vs. Tesseract) and extraction methods (Regex vs. Local LLM vs. Cloud LLM), reporting both accuracy rates and latency (`elapsed_ms`).
  - **Improvement**: `evaluate.py` must support an `--all-methods` / `--all-engines` matrix run or persist distinct target files (e.g., `results/metrics_regex_easyocr.csv`, `results/metrics_llm_easyocr.csv`), culminating in an integrated multi-method comparative summary table inside `results/metrics.md`.

### 2.2 Golden Set Check Utility (`scripts/check_golden_set.sh`)
- **Current State**: Implemented (commit `3d4c7805`).
- **Implementation**: Created `scripts/check_golden_set.sh` with strict POSIX flags (`set -euo pipefail`), invoking `python -m src.golden_set` and `python -m src.dataset_stats`. Validates all 27 records and outputs corpus stratification cleanly.

### 2.3 Presentation Deck (`Docs/Presentation/`)
- **Current State**: WP-18 remains pending.
- **Improvement**: Create the slide deck following the 6-slide structure specified in `implementation_plan.md` Section 8.6 (corpus motivation, stratification criteria, capture protocol, ground-truth verification, corpus statistics, and limitations).

---

## 3. Dataset and Ground-Truth Quality Vulnerabilities (`data/ground_truth.csv`)

A rigorous inspection of `data/ground_truth.csv` against `src/normalizers.py` revealed two critical issues that will skew benchmark metrics if not corrected:

### 3.1 Merchant `entidade` Mismatches (Snake_case Slugs vs. Printed Headers)
- **Defect**: In `data/ground_truth.csv`, values in the `entidade` column were populated using filename slugs rather than the actual merchant names printed on the receipts (e.g., `continente_amadora`, `jrn_doçarias`, `bcm_bricolage`, `prod_lavoura`, `alentejo_ubbo`, `vecc_veterinary`, `gelataria_versailles`).
- **Root Cause & Impact**: `valores_equivalentes("entidade", esperado, obtido)` runs `normalizar_entidade()`, which performs accent and corporate suffix removal, followed by an exact string equality check (`expected_value == obtained_value`). On physical receipts, merchant headers read `"MODELO CONTINENTE HIPERMERCADOS S.A."`, `"CONTINENTE BOM DIA"`, `"O DAS BIFANAS, LDA"`, or `"VERSAILLES"`. Comparing `"continente amadora"` against `"modelo continente hipermercados"` results in `acertou=False`. As a result, both the Regex baseline and LLM extractors will artificially fail on merchant name extraction across most test documents.
- **Remedy**: Re-annotate the `entidade` column in `data/ground_truth.csv` to reflect the trading names as physically printed on the documents, adhering to `implementation_plan.md` Section 8.4 (*"Valores exatamente como impressos"*).

### 3.2 File Naming and Pair Labeling Inconsistencies
- **Defect**: Crumpled receipts are named `NN_amachucado_...` on disk, whereas the CSV and code contracts strictly require `grupo="amarrotado"`. Furthermore, none of the files currently carry the `_par` suffix specified in `implementation_plan.md` Section 8.2 to formally denote paired flat/crumpled captures.
- **Remedy**: Align dataset file naming or document the convention explicitly, ensuring paired receipts are identifiable in corpus reporting.

---

## 4. Detailed Module Audit and Technical Improvements

*Note: The technical improvements detailed below across sections 4.1 to 4.17 were implemented, tested, and integrated into `master` in commits `66209907`, `3d4c7805`, and `8ae84b28`.*

### 4.1 Data Contracts (`src/types.py`)
- **Loose Types in Dataclasses**:
  - `OcrEngineName`, `MethodName`, `GroupName`, `FieldName`, and `ProviderName` `Literal` aliases are defined (lines 21-44) but not applied inside the dataclasses. Specifically, `OcrDocument.engine` is `str`, `Extraction.method` is `str`, `MatchResult.grupo` is `str`, `MatchResult.campo` is `str`, and `Answer.provider` is `str`.
  - **Improvement**: Enforce the `Literal` types directly on dataclass attributes for full type-checker enforcement under Pyright.
- **Geometric Invariant Validation in `BBox`**:
  - `BBox` does not verify that `x2 >= x1` and `y2 >= y1`. A negative-dimension box can be constructed without validation errors.
  - **Improvement**: Add a `__post_init__` guard validating non-negative dimensions and coordinate ordering.

---

### 4.2 Application Settings (`src/settings.py`)
- **Global Environment Mutation in `load_settings`**:
  - Lines 88-89 invoke `load_dotenv(dotenv_path=env_file, override=False)`. If a test executes `load_settings` pointing to an isolated directory with an `.env` file, `load_dotenv` permanently mutates `os.environ` for all subsequent tests in the same process.
  - **Improvement**: Use `dotenv.dotenv_values(env_file)` merged with `os.environ` to keep isolated `Settings` instances from leaking environment variables across tests.
- **Unlogged Engine Typos**:
  - Line 105: `"tesseract" if raw_engine == "tesseract" else "easyocr"`. Any typo (e.g., `OCR_ENGINE=tessract`) silently defaults to EasyOCR without warning.
  - **Improvement**: Log a warning when an unrecognized engine name is encountered in the environment.
- **Missing Knobs**:
  - Missing configurations for `max_side`, `auto_deskew`, `auto_orient_osd`, and local LLM runtime parameters (`LLAMA_PORT`, `LLAMA_NGL`, `LLAMA_CTX`).

---

### 4.3 Pure Field Normalizers (`src/normalizers.py`)
- **Documentation and Style Rule Violations**:
  - The module contains Portuguese docstrings (`Argumentos:`, `Retorna:`, `Retorno:`), a typo in the module docstring (`"""Funções  para..."""`), an incorrect parameter name in `valores_equivalentes` (`adquirido:` instead of `obtido:`), and lacks the required `Ref:` line.
  - **Improvement**: Translate all docstrings to Google-style English and add the `Ref:` header.
- **Early-Exit Bug in `parse_data`**:
  - Inside the pattern iteration loop (lines 121-124):
    ```python
    try:
        return date(year, month, day)
    except ValueError:
        return None
    ```
    If a line contains an 8-digit number (such as an order reference `99999999`) preceding an actual date, pattern #3 (`compact`) matches `99999999`, raises `ValueError`, and **immediately returns `None`**, terminating the function before pattern #4 (`DD/MM/YYYY`) is evaluated.
  - Furthermore, `re.search` evaluates only the first occurrence per pattern. If a line contains `"Ref: 12-34-56 Data: 26-10-27"`, the first match fails validation, returning `None` without evaluating the remainder of the line.
  - **Improvement**: Replace `return None` with `continue`, and utilize `re.finditer` to test all candidate tokens.
- **Ambiguity in 2-Digit Hyphenated Dates**:
  - `parse_data` unconditionally parses `(\d{2})-(\d{1,2})-(\d{1,2})` as `YY-MM-DD`. In Portugal, receipts frequently print `DD-MM-YY` (e.g., `27-10-26`).
  - **Improvement**: If the first number is greater than 12 and the third number is less than or equal to 31, and `2000 + first` exceeds `date.today().year`, treat the format as `DD-MM-YY`.
- **First-Number Extraction Limit in `parse_moeda`**:
  - `parse_moeda` matches the first numeric sequence in a string. When given lines such as `"2026-10-27 16,10"` or `"ART 100 16,10"`, it parses the leading number (`2026` or `100`).
  - **Improvement**: Add an optional `prefer_last: bool = False` parameter, or let caller modules specify whether to target the closing numeric token.
- **Overly Restrictive Suffix Removal in `normalizar_entidade`**:
  - Line 179: `entity_suffixes = {"lda", "sa", "unipessoal"}` removes words only if they are the final token. If a merchant name is `"JRN DOCARIAS, SOCIEDADE UNIPESSOAL LDA"`, `"sociedade"` is retained.
  - Furthermore, because `extractor_regex.py` reuses `normalizar_entidade` as its internal `_fold()` function, single-word merchants ending in `"sa"` or `"lda"` have those letters stripped.

---

### 4.4 Golden Set Loader and Validator (`src/golden_set.py`)
- **Coupling to Global Settings in `carregar_ground_truth`**:
  - Line 168: `records, errors = _validate_csv(path, get_settings().receipts_dir)`.
  - `carregar_ground_truth` does not accept `receipts_dir: Path | None = None`. Tests and CLI overrides must monkeypatch `src.golden_set.get_settings` to override the receipts directory.
  - **Improvement**: Add `receipts_dir: Path | None = None` to `carregar_ground_truth`, defaulting to `get_settings().receipts_dir` when omitted.
- **Missing Check for Final Consumer Sentinel (`999999990`)**:
  - `validar_nif("999999990")` returns `True`. If an annotator inputs `999999990` into `data/ground_truth.csv`, validation passes, creating a mismatch with extractors that map `999999990` to `AUSENTE`.
  - **Improvement**: Add explicit validation in `golden_set.py` rejecting `999999990` in `nif_cliente` or `nif_comerciante` and prompting the user to leave the cell blank.
- **Missing Date Lower-Bound Check**:
  - Validates `parsed_date <= today`, but does not verify `parsed_date >= EARLIEST_PLAUSIBLE_DATE` (`2000-01-01`).

---

### 4.5 Image Preprocessing Pipeline (`src/preprocessing.py`)
- **Missing `tesseract_cmd` Forwarding**:
  - Line 620: `preprocess()` invokes `detect_orientation_osd(filtered_bgr, min_confidence=3.0)` without forwarding `tesseract_cmd`. On Windows systems where Tesseract is located outside `PATH` and specified in `.env`, the probe fails silently.
  - **Improvement**: Pass `tesseract_cmd` through `PreprocessOptions` into `detect_orientation_osd`.
- **State Inconsistency in Intermediate Steps Dictionary**:
  - In lines 578-585, when `did_crop` is true, `steps["rodada"]` is assigned `cropped.copy()`, meaning the rotation step contains an already cropped image.
  - Furthermore, when 180-degree flip correction triggers via QR heuristic or OSD (lines 615-633), `steps["recortada"]` and `steps["filtrada"]` are rotated, but `steps["rodada"]` remains unrotated.
  - **Improvement**: Ensure each intermediate step in `steps` strictly reflects the true state of that processing stage.
- **Static Kernel Size in Shadow Attenuation**:
  - `attenuate_shadows(gray, kernel_size=25)` uses a static 25x25 structuring element. For large bold header store logos whose stroke width exceeds 25 pixels, morphological closing fails to bridge the character strokes, causing background division to wash out the interior of store name text.
  - **Improvement**: Dynamically scale `kernel_size` relative to image width or blend with the source luminance.
- **Literal Windows Backslash Path**:
  - Line 503 hardcodes `Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")`, violating cross-platform style guidelines. Replace with `Path("C:/Program Files/Tesseract-OCR/tesseract.exe")`.

---

### 4.6 EasyOCR Engine (`src/ocr_engine.py`)
- **Reading-Order Fragmentation on Multi-Column Lines**:
  - EasyOCR frequently breaks a single horizontal receipt line into two separate bounding boxes: e.g., `"TOTAL"` on the left (`y1=502`) and `"16,10 EUR"` on the right (`y1=500`).
  - Because `y1=500` is slightly higher, `"16,10 EUR"` is sorted *before* `"TOTAL"`, and each fragment is placed into a separate `OcrLine`.
  - Consequently, `extractor_regex.py` fails to match `_TOTAL_LINE` (`r"\bTOTAL...(?P<amount>...)$"`), producing a false negative.
  - **Improvement**: Post-process raw OCR detections by grouping boxes whose vertical overlap or centroid distance falls within a threshold (e.g., 50% of box height) into a unified `OcrLine`, sorting horizontally by `x1`.

---

### 4.7 Tesseract Engine (`src/ocr_tesseract.py`)
- **Documentation and Error String Violations**:
  - Portuguese docstrings and error messages (`Argumentos:`, `Retorno:`, `Exceções:`, `"A imagem de entrada não pode estar vazia..."`) violate the required English Google-style convention and contract error message (`"Input image must not be empty or zero-sized."`).
  - **Improvement**: Standardize docstrings and error strings to English and add the `Ref:` header.
- **Missing Dependency Injection**:
  - Lacks an injection point for testing (e.g., `image_to_data_fn=None`), requiring tests to monkeypatch the module.

---

### 4.8 Portuguese AT QR Code Reader (`src/qr_reader.py`)
- **Masking by Promotional / Loyalty QR Codes**:
  - Receipts from large chains (e.g., Continente, Pingo Doce, McDonald's) often print two QR codes: a promotional app URL and the fiscal AT QR code.
  - In `_decode_single_image()` (lines 98-109), `pyzbar.decode(img)` returns the first decoded text. If the first code is a promotional URL, `parse_at_qr_payload()` returns `None`, and the function exits without evaluating the second detected barcode.
  - **Improvement**: Loop through all detected barcodes in `pyzbar.decode` and `detectAndDecodeMulti`, validating each with `parse_at_qr_payload` until a valid AT payload is encountered.
- **Redundant Processing Stages**:
  - Stage 1 tests BGR and Stage 2 tests Grayscale. Because `preprocess()` outputs an aligned image whose 3 BGR channels are already identical grayscale luminance, Stage 1 and Stage 2 execute duplicate work.
- **Static Candidate Blob Area Bounds**:
  - Line 134 hardcodes `10000 <= area <= 700000`. On downscaled images, small QR codes can have an area under 10000 pixels.
  - **Improvement**: Define area thresholds as proportions of total image area (`0.01 * total_area` to `0.60 * total_area`).

---

### 4.9 Unified LLM Client (`src/llm_client.py`)
- **Lack of Multimodal VLM Capabilities (Approach C)**:
  - `PROJECT_PLAN.md` (Level 2, Approach C) outlines direct multimodal VLM extraction without prior OCR. `LLMClient.complete` currently accepts only text prompts (`system: str, user: str`).
  - **Improvement**: Add an optional `image: np.ndarray | None = None` parameter to `LLMClient.complete` to enable multimodal experimentation.
- **Disproportionate Cloud Timeout**:
  - `build_client` sets `timeout_s=cfg.llama_timeout_s` (120 seconds) for OpenAI, while Gemini has no timeout set. If cloud APIs hang during a live presentation, the UI remains unresponsive for two minutes before local failover activates.
  - **Improvement**: Apply a fast timeout (e.g., 15-20 seconds) for cloud backends so failover to `llama-server` is responsive.

---

### 4.10 Regex Baseline Extractor (`src/extractor_regex.py`)
- **Omission of Valid Portuguese Receipt Total Formats**:
  - Line 77: `_TOTAL_LINE` requires the amount to be placed at the very end of the line following `TOTAL`, rejecting common receipt layouts:
    - Currency prefix: `"TOTAL EUR 16,10"` or `"TOTAL € 16,10"`
    - Semantic qualifiers: `"TOTAL LIQUIDO 16,10"`, `"TOTAL GERAL 16,10"`, `"TOTAL A PAGAR 16,10"`
    - Action phrases: `"A PAGAR: 16,10"`, `"VALOR A PAGAR 16,10"`
  - **Improvement**: Permit optional qualifiers and leading currency markers while preserving end-of-line boundaries to avoid false positives on `"TOTAL IVA"` or `"SUBTOTAL"`.
- **Top-of-Receipt Banner False Positives**:
  - Receipts often open with greeting phrases (`"BEM-VINDO"`, `"OBRIGADO PELA VISITA"`). Adding `"bem"`, `"vindo"`, `"copia"`, `"segunda"`, `"via"` to `_ENTITY_STOPWORDS` avoids misclassifying headers as trading names.

---

### 4.11 LLM Structured Extractor (`src/extractor_llm.py`)
- **Top-to-Bottom Grounding Vulnerability for `total`**:
  - In `_locate_line` (lines 358-366), when searching for the line matching `total`, the code scans lines top-to-bottom and returns the first substring match.
  - On single-item receipts where article price, subtotal, and total are identical (e.g., `8.05`), top-to-bottom scanning anchors the total to the article line at the top rather than the `TOTAL` line at the bottom.
  - **Improvement**: Traverse lines in reverse (`reversed(lines)`) and prioritize lines containing the keyword `"total"`.
- **String Substring False Matches**:
  - Checking `target in candidate` for `"8.05"` matches inside numbers like `"18.05"` or `"28.05"`. Use boundary-aware token splitting.

---

### 4.12 Visual Evidence Rendering (`src/evidence.py`)
- **Full-Receipt Zoom in Multi-Line `crop()`**:
  - In `src/app.py` (lines 220-231), `process_document` passes line IDs for all 5 extracted fields (spanning from line 1 at the top to line 40 at the bottom) to `crop()`.
  - Because `crop()` computes the union bounding box, the resulting evidence snippet spans the entire receipt, defeating the purpose of a focused visual crop.
  - **Improvement**: In `src/app.py`, allow the user to select an individual field row in `tbl_fields` to crop that field's evidence, or have `crop()` construct a stacked montage of the cited lines when their vertical span exceeds 30% of image height.

---

### 4.13 Q&A Assistant (`src/qa_assistant.py`)
- **Punctuation and Stopword Contamination in `_locate_supporting_lines`**:
  - Lines 120-127 split `f"{question} {answer}"` on spaces without stripping punctuation. Tokens like `"16,10."` or `"talão?"` retain trailing characters and fail substring checks.
  - Stopwords with length >= 3 (`"qual"`, `"valor"`, `"para"`, `"com"`, `"loja"`) are included in the search keywords. If an LLM hallucinates `status="ENCONTRADO"` on missing data, any line containing `"loja"` or `"valor"` gets matched, incorrectly validating the hallucinated answer.
  - **Improvement**: Strip punctuation, exclude common question stopwords, and prioritize answer-specific tokens (digits, proper nouns).
- **Sub-string Bug in `is_found` Check**:
  - Line 173: `is_found = "ENCONTRADO" in status_raw`.
  - If a model responds with `"NAO ENCONTRADO"` or `"NÃO ENCONTRADO"`, `"ENCONTRADO" in status_raw` evaluates to `True`.
  - **Improvement**: Check `status_raw == "ENCONTRADO"`.
- **Incomplete Test in `tests/test_qa_assistant.py`**:
  - `test_main_cli_preview` (lines 117-130) wraps execution in `with contextlib.suppress(Exception):` without assertions or mocks, risking invocation of live EasyOCR/LLM dependencies during offline test runs.
  - **Improvement**: Inject mocks and assert on stdout.

---

### 4.14 Scoring and Aggregation (`src/scoring.py`)
- **Language and Style Violations**:
  - Mixed Portuguese and English docstrings, a typo (`"groud-truth"` on line 47), and a missing `Ref:` line.
  - **Improvement**: Unify docstrings into English and add the `Ref:` header.
- **Ignoring Extractor Field Status**:
  - In `pontuar_registo`, comparisons check `valores_equivalentes(field, expected, obtained)` purely against `.value`. If an extractor sets `FieldValue(value="16.10", status="INVALIDO")`, but the raw string parses to the expected total, it is scored as `acertou=True`.
  - **Improvement**: A field marked `INVALIDO` or `AUSENTE` should never score `acertou=True` when `expected is not None`.

---

### 4.15 Corpus Statistics (`src/dataset_stats.py`)
- **Untyped Exception Access in `main()`**:
  - Lines 131-134 access `exc.args[0]` rather than the typed `exc.errors` attribute of `GoldenSetError`.
- **Receipts Directory Override Failure**:
  - `carregar_ground_truth(args.csv)` delegates to `get_settings().receipts_dir`, ignoring `settings_override.receipts_dir`.

---

### 4.16 Gradio Interactive UI (`src/app.py`)
- **Selector Mismatch in Q&A Assistant**:
  - Tab 1 provides a single radio button `extractor_choice` defaulting to `"Regex Baseline"`.
  - This same selector is passed as `provider_choice` to `answer_question` (line 541).
  - Inside `answer_question`, `"Regex Baseline"` maps to `"llm_local"`. If `llama-server` is not running, clicking `"Perguntar ao Documento"` fails with a connection error even though the extraction above it succeeded via Regex.
  - **Improvement**: Decouple the Q&A LLM provider selector from the structured extraction method selector, or display an explicit UI alert when local LLM inference is unreachable.
- **Table Formatting and UTF-8 BOM Handling in Tab 3**:
  - `load_benchmark_report` opens `metrics.csv` with `encoding="utf-8"`. Since `scoring.py` writes `metrics.csv` with `encoding="utf-8-sig"`, the first column header becomes `"\ufefffilename"`.
  - In `_refresh_benchmark`, column headers are discarded and only raw data rows are passed to `tbl_benchmark`, causing Gradio to display numeric column indices (`0, 1, 2...`) instead of actual field names.
  - **Improvement**: Read with `encoding="utf-8-sig"` and pass both headers and rows to `gr.Dataframe`.
- **Thread Safety on Shared Caches**:
  - `_OCR_CACHE` and `_ENGINE_CACHE` are mutated across Gradio worker threads without a `threading.Lock()`.
- **Visual Discrepancy Indicators for AT QR Cross-Validation**:
  - While Tab 1 displays extracted values alongside QR values, it does not visually highlight discrepancies (e.g., when OCR misreads a digit that decoded cleanly in the QR payload).

---

### 4.17 Server Start Scripts (`scripts/run_llama_server.{sh,ps1,bat}`)
- **Configuration Drift Against Documentation**:
  - `AGENTS.md` (line 75), `README.md` (line 279), and `implementation_plan.md` (line 440) state that default execution uses:
    `models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` with `NGL=0`.
  - The scripts on disk still default to:
    `models/Qwen2.5-3B-Instruct-Q8_0.gguf` with `NGL=99`.
  - On CPU-only systems or laptops with small GPUs, running the scripts without overrides fails due to missing model files or VRAM allocation errors.
  - **Improvement**: Update default variable values across `.sh`, `.ps1`, and `.bat` to match the documented defaults (`models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` and `NGL=0`).

---

## 5. Prioritized Implementation Roadmap

### 5.1 Completed Work
- **Core Pipeline & Robustness Bug Fixes (Sections 4.1 to 4.17)**:
  - `src/types.py`: Literal type annotations and BBox non-negative ordering validation.
  - `src/settings.py`: Isolated `dotenv_values` parsing, engine warning logging, and runtime knobs.
  - `src/normalizers.py`: English Google docstrings with `Ref:`, `re.finditer` in `parse_data`, and multi-word legal suffix stripping in `normalizar_entidade`.
  - `src/golden_set.py`: `receipts_dir` parameter, `999999990` sentinel rejection, and `2000-01-01` date lower bound.
  - `src/preprocessing.py`: `tesseract_cmd` forwarding, intermediate steps dictionary consistency, and forward-slash Windows paths.
  - `src/ocr_engine.py`: Baseline-aware horizontal bounding-box fragment merging.
  - `src/ocr_tesseract.py`: Standardized English docstrings/errors with `Ref:` and `image_to_data_fn` injection hook.
  - `src/qr_reader.py`: Multi-QR barcode loop (`_decode_all_codes`) preventing promotional code masking and proportional blob area bounds.
  - `src/llm_client.py`: Multimodal image parameter support and removal of `llm_openai` direct provider.
  - `src/extractor_regex.py`: Expanded `_TOTAL_LINE` qualifiers/currencies and greeting stopwords.
  - `src/extractor_llm.py`: Reverse-order total line grounding with keyword prioritization.
  - `src/qa_assistant.py`: Punctuation/stopword token extraction and strict `status_raw == "ENCONTRADO"` validation.
  - `src/scoring.py`: Standardized English docstrings with `Ref:` and `INVALIDO`/`AUSENTE` status check.
  - `src/dataset_stats.py`: Typed `exc.errors` access and `receipts_dir` forwarding.
  - `src/app.py`: Decoupled `qa_provider_choice`, `utf-8-sig` CSV reading with table headers, and `_CACHE_LOCK` thread safety.
  - `scripts/run_llama_server.*`: Default model aligned with `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` and `NGL=0`.
  - `scripts/check_golden_set.sh`: Automated POSIX check script for golden set and dataset stats.

### 5.2 Remaining Deliverables
1. **Phase 1: Dataset Integrity & Annotations (WP-17)**
   - Re-annotate the `entidade` column in `data/ground_truth.csv` with the actual printed merchant names (e.g. replacing filename slugs like `continente_amadora` with `"MODELO CONTINENTE HIPERMERCADOS S.A."`).
   - Run `bash scripts/check_golden_set.sh` and verify zero validation errors.

2. **Phase 2: Benchmark Runner Implementation (WP-15)**
   - Implement `src/evaluate.py` according to `implementation_plan.md` Section 6 (WP-15), supporting automated output generation for `results/metrics.csv` and `results/metrics.md`.
   - Add unit and contract tests in `tests/test_evaluate.py` using offline test doubles.

3. **Phase 3: Presentation Deck (WP-18)**
   - Build the slide deck in `Docs/Presentation/` following the 6-slide structure in `implementation_plan.md` Section 8.6.

4. **Phase 4: Final Verification & Benchmark Run**
   - Run the full verification pipeline:
     ```bash
     uv run ruff format --line-length 120 src tests
     uv run ruff check src tests
     uv run pyright src tests
     uv run pytest
     ```
   - Execute the official benchmark:
     ```bash
     uv run python -m src.evaluate
     ```
