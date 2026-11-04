# Development Best Practices & Engineering Handbook

A consolidated reference of software engineering patterns, architectural invariants, toolchain standards, and testing conventions established across ISEG executive master group projects (`CVGAI-6ed-group-4`, `nlpg-6ed-group-4`, and `tsf-6ed-group-4`).

---

## 1. Toolchain & Environment Management

### Package Management with `uv`
- **Use `uv` exclusively**: Never invoke raw `pip`, `poetry`, or `conda`. Environment synchronization (`uv sync`), dependency addition (`uv add`), and script execution (`uv run`) must remain deterministic.
- **Python Version Locking**: Pin supported Python versions explicitly in both `pyproject.toml` (`requires-python`) and `.python-version`. When machine learning dependencies (e.g. PyTorch, EasyOCR) lack stable wheels for newer minor releases, lock strictly to the verified version (e.g. `>=3.12,<3.13`).
- **Execution Mode (`package = false`)**: For academic research and evaluation pipelines, configure `[tool.uv] package = false`. Run modules as top-level Python modules from the repository root:
  ```bash
  uv run python -m src.<subpackage>.<module>
  ```
  Never install the active project as an editable distribution (`pip install -e .`). This eliminates stale build artifacts, editable link drift, and namespace collisions.
- **Binary & Native Library Discipline**:
  - Always install headless graphics builds (e.g. `opencv-python-headless`). Never co-install GUI/contrib distributions (`opencv-python`, `opencv-contrib-python`), as multiple wheels claiming the same `cv2` namespace corrupt the C-extension runtime.
  - Rely on native standard libraries for path handling (`pathlib.Path`), never raw string concatenation or OS-specific separators (`\`).
  - Support UTF-8 across operating systems (set `$env:PYTHONUTF8 = "1"` on Windows).

---

## 2. Code Quality & The 4-Step Verification Gate

Every change must pass the four-stage verification pipeline before committing:

```bash
# 1. Code Formatting (120-character line ceiling)
uv run ruff format --line-length 120 src tests

# 2. Fast Static Linting (Style, imports, bug patterns)
uv run ruff check src tests

# 3. Static Type Checking (Strict boundary validation)
uv run pyright src tests

# 4. Deterministic Offline Test Suite
uv run pytest
```

### Static Analysis Invariants
- **Line Length**: Set code line limit to 120 characters in `pyproject.toml` under `[tool.ruff]`. Allow comments/docstrings flexibility via `ignore = ["E501"]` while enforcing code structure through the formatter.
- **Type Checker**: Use standalone `pyright` in `typeCheckingMode = "standard"`. Reject untyped signatures on all public functions, classes, and return types. Avoid `Any` except where third-party SDKs omit stubs.
- **No Emojis**: Maintain a clean, professional, human engineering tone. Absolutely zero emojis across code, docstrings, tests, CLI output, markdown documentation, commit messages, and PRs.

---

## 3. Architecture & Modularity

### Module Size & Human Cognitive Limits
- **The 200–300 Line Sweet Spot**: Files exceeding 400 lines induce scroll fatigue and cognitive overload. Keep self-contained domain modules between 150 and 250 lines, with a hard ceiling of 300 lines.
- **Decompose by Domain Responsibility, Never by Language Construct**:
  - **Good**: Grouping related algorithms by domain: `geometry.py` (contour/quad warping), `photometry.py` (shadows, CLAHE, deskew), `orientation.py` (flip detection, OSD).
  - **Anti-Pattern**: Java-style decomposition into single-class files (`PreprocessOptions.py`, `PreprocessResult.py`) or creating generic junk drawers (`classes/`, `utils.py`, `auxiliary.py`).
- **The Facade Pattern for Public Stability**: When decomposing monolithic files, maintain the original module as a backward-compatible facade:
  - Re-export all public functions, classes, and backward-compatible aliases.
  - Declare an explicit `__all__` list.
  - Ensure external callers and tests do not require import rewrites.

### Keystone Data Contracts (`src/core/types.py`)
- **Centralize Pipeline Types**: Define all cross-cutting domain data structures in a single keystone file (`src/core/types.py`). Pipeline stages (preprocessing, OCR, extraction, evaluation, UI) exchange only these types.
- **Immutability & Slot Efficiency**: Use `@dataclass(frozen=True, slots=True)` for data exchange transfer objects (`BBox`, `OcrLine`, `FieldValue`, `Extraction`, `GroundTruthRecord`, `MatchResult`).
- **Runtime Enums via Literals**: Pair strict `Literal` types with runtime tuple constants to validate CLI arguments and UI choices:
  ```python
  Status = Literal["ENCONTRADO", "AUSENTE", "INVALIDO"]
  STATUSES: tuple[Status, ...] = ("ENCONTRADO", "AUSENTE", "INVALIDO")
  ```
- **Semantic Data Distinctions**:
  - Represent missing fields as `None` (in ground truth) and `"AUSENTE"` (in extraction status). Never use empty strings, `0`, or `"N/A"` to signal absence.
  - Distinguish structurally invalid values (`"INVALIDO"`) from omissions (`"AUSENTE"`).

### Centralized Settings (`src/core/settings.py`)
- **Single Source of Truth**: Isolate all environment reads, path resolution, and default hyperparameters in `src/core/settings.py`.
- **Zero Environment Mutation**: Load environment variables via `dotenv.dotenv_values()` rather than mutating `os.environ`.
- **Caching Discipline**: Use `@lru_cache` for `get_settings()` in production, but provide `load_settings(repo_root=...)` for test injection.

---

## 4. Pure Functions & Side-Effect Isolation

### Immutability in Image Processing
- **Array Validation**: Every public function receiving an image must validate non-empty dimensions (`validate_image(image)`).
- **Non-Destructive Operations**: Never mutate input arrays in place. Always return a newly allocated or `.copy()` array.

### Pure Normalization Layer (`src/core/normalizers.py`)
- Normalizers (NIF validation, date parsing, currency normalization, string cleaning) must be 100% pure functions: zero filesystem I/O, zero network calls, zero dependence on global state.
- Make normalization routines idempotent and tolerant of OCR noise (e.g. garbled country prefixes, spacing artifacts).

### Thread-Safe In-Memory Caching
- Guard shared mutable state (document caches, engine instances) with explicit `threading.Lock()` primitives.
- Use deterministic cryptographic cache keys (`hashlib.sha256`) incorporating raw bytes and hyperparameter values.
- Bound memory usage with LRU-style eviction (e.g. popping oldest keys when cache size exceeds limits).

---

## 5. Testing Philosophy & Test Seams

### The Five Invariants of Fast Unit Testing
1. **100% Offline & Deterministic**: Unit tests must never initiate HTTP/HTTPS requests, require running database/model servers, or download remote weights.
2. **Sub-Second Execution**: Standard unit tests must execute in milliseconds using synthetic images (`np.ndarray`), mocked documents, or simulated payloads.
3. **No Commits to Data Directories**: Temporary files must be created strictly in pytest's `tmp_path` fixture. Tests must never modify `data/` or commit ground truth artifacts.
4. **Isolated Slow Markers**: Live model tests (EasyOCR real inference, GPU sweeps) belong behind `@pytest.mark.slow` and must be deselected by default in `pyproject.toml` (`addopts = "-q --strict-markers -m 'not slow'"`).
5. **Preserve Test Seams during Refactoring**:
   - Never hoist function-local imports (`build_client`, internal extraction helpers) to module level if unit tests monkeypatch them at runtime.
   - Do not move functions whose module-level bindings are patched by tests without updating the corresponding test seams.

### Dependency Injection over Monolithic Mocking
- Design components with constructor injection hooks for testability:
  ```python
  class EasyOCREngine:
      def __init__(self, reader: Any | None = None) -> None:
          self._reader = reader or self._init_reader()
  ```
- Prefer injecting lightweight fake clients or monkeypatching callable arguments over patching complex C-extension internals.

---

## 6. Evaluation, Golden Set & Model Robustness

### Ground Truth Integrity
- **Contract Enforcement**: Provide an automated validation task (`./run check-dataset`) that verifies schema columns, image file existence, category groups, checksums, and date/total validity.
- **Corpus Balance Tracking**: Automatically compute and display dataset distributions (e.g. pristine vs crumpled documents, merchant representation) as part of baseline checks.

### Transparent Model Failover
- When deploying cloud LLMs (Gemini, OpenAI), wrap them with transparent local failover (`FailoverLLMClient`).
- Live demonstrations, benchmark sweeps, and offline test runs must degrade gracefully to local servers (e.g. `llama-server`) rather than crashing on quota (429) or connection timeouts.
- Record the answering backend in `last_effective_provider` so benchmarks reflect the actual generator.

### Strict Schema Constraining
- Local inference servers (`llama.cpp`, `llama-server`) require flat, closed JSON schemas (`additionalProperties: false`) to construct grammar constraints.
- Explicitly describe the required JSON schema inside the system prompt: small models frequently ignore external grammar definitions if the prompt text lacks explicit schema instructions.
- Sanitize schemas for cloud providers (e.g. stripping `additionalProperties` for Gemini API compatibility).

---

## 7. CLI Architecture & Application Layer

### Standard CLI Convention
Every runnable module must expose a uniform entry point:
```python
def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    ...

if __name__ == "__main__":
    main()
```
- **Deferred Heavy Imports**: Defer heavy third-party imports (Pillow, EasyOCR, PyTorch, Gradio) inside `main()` or constructors to keep standard module imports cheap and fast during test collection.
- **Zero Library Prints**: Library functions must never call `print()`. Restrict stdout/stderr output strictly to `main()` CLI functions or structured logging (`logging.getLogger(__name__)`).

### Modular UI Architecture (Gradio / Streamlit)
- **Decouple Layout from Business Logic**: Never inline pipeline processing logic inside UI block declarations. Keep Gradio layout construction in a dedicated layout module (`src/app_layout.py`) that accepts callable handler functions as arguments.
- **Thread Application Settings**: Pass explicit `Settings` instances through UI event callbacks rather than relying on global mutable state.

---

## 8. Documentation, Traceability & Git Workflow

### Traceability via Work Packages
- Every source module should carry a top-level module docstring with a `Ref:` tag mapping directly to the project implementation plan:
  ```python
  """Contour geometry, boundary detection, and perspective transformation.

  Ref: Docs/implementation_plan.md, section 7, WP-04.
  """
  ```
- Write docstrings in Google style (English) for all public functions, classes, parameters, and exceptions.

### Trunk-Based Development (TBD)
- **Short-Lived Branches**: Work in focused branches (`feat/wp-*`, `fix/*`) merged back to main frequently.
- **Target Diff Size**: Keep work package diffs focused (< 200–300 lines of net change). Review moved files using git similarity flags (`git diff -M`).
- **Synchronized Documentation**: Update `README.md`, `AGENTS.md`, and specification documents in the same commit that delivers the feature. Documentation must describe what currently exists, not what is planned.
