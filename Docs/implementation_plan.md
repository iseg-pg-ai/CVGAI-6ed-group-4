# Plano de Implementação — Leitor Inteligente de Documentos Visuais

Documento de engenharia que transforma o `PROJECT_PLAN.md` em trabalho executável. Define a
arquitetura de código, os contratos de dados, a divisão de trabalho, o protocolo de revisão e
a ordem de execução.

Complementa (não substitui) o `PROJECT_PLAN.md`, que continua a ser o documento de
enquadramento e de argumentação para o relatório.

Data de referência: 2026-10-30. As datas abaixo são relativas (Dia 0 = arranque). Ancorar ao
prazo real de submissão antes de partilhar.

Equipa: Vitor (pipeline), José (camada de dados e métricas), Raquel (corpus e narrativa).

**Estado em 2026-11-04.** WP-00, WP-01, WP-02, WP-03, WP-04, WP-05, WP-06, WP-07, WP-08, WP-09,
WP-10, WP-11, WP-12, WP-13, WP-14, WP-15, WP-16 e WP-17 estão fechados e integrados em `master`. O `src/core/settings.py`
foi acrescentado fora da numeração para centralizar o acesso ao ambiente e aos caminhos, e `./run check-dataset`
valida o dataset de forma automática. O detalhe do que foi entregue, e do que diverge do plano original, está na secção 5.1.
As restantes secções descrevem o alvo e continuam válidas para o pacote por executar (WP-18).

---

## 0. Princípios de Engenharia

1. **Contratos primeiro, implementação depois.** O ficheiro `src/core/types.py` é escrito e revisto
   antes de qualquer módulo. É isso que permite duas pessoas trabalharem em paralelo sem
   colisões.
2. **Fronteiras de responsabilidade nítidas.** Cada ficheiro tem um dono. Um pacote de
   trabalho nunca edita um ficheiro que pertence a outra pessoa.
3. **Funções puras sempre que possível.** Normalização, validação e pontuação não fazem
   entrada/saída, não leem o ambiente e não desenham nada. Testam-se com `pytest` em
   milissegundos.
4. **O Vitor é dono do pipeline visual; o José é dono da camada de dados e métricas; a Raquel
   é dona do corpus e da narrativa.** Esta divisão maximiza o contributo do José e mantém-no
   fora do caminho crítico.
5. **Tudo tem de correr offline.** Nenhum teste e nenhuma demonstração pode depender de rede
   ou de chaves de API.

---

## 1. Stack e Decisões Técnicas Fechadas

| Decisão | Escolha | Justificação |
| :--- | :--- | :--- |
| Gestor de pacotes | `uv` + `pyproject.toml` + `uv.lock` | Igual ao `nlpg-6ed-group-4`; reprodutível e rápido |
| Versão de Python | **3.12** | O EasyOCR 1.7.2 arrasta `torch`/`torchvision`; 3.12 é a combinação com mais rodagem. **Não usar 3.14** como no NLPG: os wheels de `torchvision` para 3.14 ainda são irregulares e o EasyOCR não tem versão nova desde 2024. Validar com `uv add easyocr` antes de fixar. |
| Formatação | `ruff format --line-length 120` | Convenção herdada do NLPG e do TSF |
| Verificação de tipos | `pyright` (modo standard) | Convenção herdada do NLPG |
| **Testes** | **`pytest` + `pytest-cov`** | O NLPG não tinha testes. Aqui são o mecanismo que permite ao José trabalhar de forma autónoma e que permite ao professor verificar o projeto num único comando. |
| OCR | EasyOCR (principal) + Tesseract (comparativo) | Ver `PROJECT_PLAN.md`, secção 4 |
| LLM/VLM | `llama.cpp` (`llama-server`) local + Google Gemini na cloud | Idem |
| Interface | Gradio | Idem |
| QR | `opencv-python-headless` (`cv2.QRCodeDetector`) com recurso alternativo a `pyzbar` | Sem dependência de serviços externos. Ver nota em 1.1: `contrib` não é necessário e conflitua com o EasyOCR. |

### 1.1 Dependências de sistema (fora do `uv`)

- `tesseract-ocr` e `tesseract-ocr-por` (pacotes do sistema, não do pip). Servem o backend OCR
  comparativo e a sonda de orientação do pré-processamento.
- Binário `llama-server` compilado com backend Vulkan (funciona na GPU antiga do portátil e na
  RX 9070 XT) ou descarregado dos releases oficiais.
- `zbar`, para o `pyzbar` que tenta a primeira descodificação do QR. É opcional: sem ela, o
  `qr_reader` usa só o `cv2.QRCodeDetector`, que é suficiente para o corpus.

As três estão documentadas no `README.md` com os comandos de instalação por distribuição, porque
o José precisa deles para correr o pipeline localmente. Nenhuma delas é necessária para a suite
de testes, que corre offline e sem binários externos.

**Nota sobre o OpenCV (validada no Dia 0).** O `easyocr` 1.7.2 fixa `opencv-python-headless`
como dependência. Instalar em paralelo `opencv-contrib-python` deixa duas distribuições a
disputar o mesmo módulo `cv2` em `site-packages`, o que produz importações que falham de forma
intermitente e difíceis de diagnosticar. Ficamos então com `opencv-python-headless`, depois de
confirmar que a build headless do OpenCV 5 expõe `cv2.QRCodeDetector` (e até `cv2.aruco`), ou
seja, sem perda de funcionalidade para o leitor de QR da AT. A resolução de dependências para
Python 3.12 foi verificada e fixa `torch==2.15.0` e `torchvision==0.29.2`.

### 1.2 pytest — configuração e execução

O `pytest` faz parte do projeto desde o primeiro commit. Não é opcional e não é uma decisão
adiada: é a forma como o José valida o próprio trabalho antes de pedir revisão e como o
professor confirma que o projeto funciona.

**Configuração em `pyproject.toml` (criada no WP-00, estado atual):**

```toml
[project]
name = "cvgai-6ed-group-4"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "ruff>=0.7",
    "pyright>=1.1.380",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q --strict-markers -m 'not slow'"
markers = [
    "slow: tests that load a real OCR engine or model (deselected by default in CI)",
]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "standard"
pythonVersion = "3.12"
```

Notas sobre estas opções:

- `requires-python = ">=3.12,<3.13"` fecha o teto superior: o EasyOCR 1.7.2 arrasta
  `torch`/`torchvision` e não há wheels estáveis acima de 3.13.
- `testpaths = ["tests"]` evita que o `pytest` ande a varrer `class-material/`, `data/` e
  `.venv/`. Sem isto, a recolha de testes tenta importar ficheiros que não são testes e a
  suite falha por motivos errados.
- `pythonpath = ["."]` põe a raiz do repositório no caminho de importação, para que os testes
  possam fazer `from src.core.normalizers import parse_moeda` sem instalar o pacote.
- `--strict-markers` faz falhar um teste que use um marcador não declarado. É a forma de
  impedir que um `@pytest.mark.skip` mal escrito passe silenciosamente.
- `-m 'not slow'` mantém o teste ao vivo do EasyOCR fora da execução por omissão, para que a
  suite continue offline e abaixo de um segundo. `uv run pytest -m slow` corre-o de propósito.
- `ruff` e `pyright` são configurados no mesmo ficheiro; a linha de comando só repete a
  `line-length` por hábito, o valor já vem da configuração.

**Comandos do dia a dia:**

```bash
uv sync                                  # instala dependências, incluindo o grupo dev
uv run pytest                            # corre a suite toda (exclui os testes 'slow')
uv run pytest tests/test_evidence.py -q             # um ficheiro
uv run pytest -k "nif" -q                           # filtra por nome
uv run pytest -m slow                               # teste ao vivo do EasyOCR
uv run pytest --cov=src --cov-report=term-missing   # com cobertura
```

**Regras a que os testes têm de obedecer:**

1. Um teste nunca acede à rede, a uma GPU, a uma chave de API ou a um modelo descarregado.
2. Um teste nunca depende da ordem de execução nem do estado deixado por outro teste.
3. Tudo o que é temporário vai para `tmp_path`; nada escreve dentro do repositório.
4. Os testes de OCR não correm o EasyOCR real. Validam o contrato (`OcrDocument`, caixas
   dentro dos limites da imagem) com imagens sintéticas e um leitor injetado. O único teste
   que carrega pesos reais está marcado com `slow` e é deselecionado por omissão.
5. Um teste que falha tem de dizer o que se esperava e o que se obteve. `assert resultado ==
   esperado` sem contexto é uma perda de tempo para quem revê.

**Fixtures partilhadas em `tests/conftest.py`** (estado atual):

```python
@pytest.fixture
def repo_root() -> Path:
    """Caminho absoluto da raiz do repositório (o que contém o pyproject.toml)."""


@pytest.fixture
def receipts_tmp(tmp_path: Path) -> Path:
    """Cria um diretório de imagens com ficheiros vazios para validação do golden set."""


@pytest.fixture
def ground_truth_valido(tmp_path: Path) -> Path:
    """Escreve um CSV mínimo mas válido, coerente com receipts_tmp."""


@pytest.fixture
def extracao_vazia() -> Extraction:
    """Extraction com os cinco campos em AUSENTE, para testes de pontuação."""


@pytest.fixture
def extracao_exemplo() -> Extraction:
    """Extraction completa, com os cinco campos encontrados e line_id."""


@pytest.fixture
def imagem_branca() -> np.ndarray:
    """Imagem BGR sintética, para os testes de visão nunca tocarem em data/receipts."""
```

**Arranque verde:** o WP-00 inclui `tests/test_smoke.py`, que importa `src.core.types`, constrói um
`BBox`, verifica a imutabilidade dos contratos e a formatação do prompt. Serve para provar que o
ambiente, a configuração do `pytest` e o caminho de importação estão corretos antes de existir
lógica de negócio, e cresceu para oito testes sem deixar de ser a primeira coisa a correr. Um
`pytest` que falha no Dia 0 é um problema de ambiente; um que falha no Dia 8 é um problema de
código, e nessa altura já não há tempo para distinguir os dois.

**Para quem avalia:** numa máquina já preparada, a suite completa corre com um único comando
(`uv run pytest`), sem rede e sem modelos. O resultado é determinístico.

---

## 2. Estrutura do Repositório

```text
CVGAI-6ed-group-4/
├── Docs/
│   ├── assignment.pdf
│   ├── implementation_plan.md
│   └── Presentation/              # mini-apresentação da Raquel (pendente)
├── data/
│   ├── receipts/                  # 27 imagens presentes (01_... a 27_...)
│   │   ├── 01_direito_controlauto.jpg
│   │   ├── 02_amachucado_tiendaanimal.jpg
│   │   └── ...
│   └── ground_truth.csv           # 27 registos validados (WP-17 concluído)
├── models/                        # .gguf (git-ignored)
├── results/                       # saídas do benchmark (versionadas, pequenas)
├── src/
│   ├── __init__.py                # versão do pacote (0.1.0)
│   ├── app.py                     # Gradio                      [Vitor]  feito
│   ├── core/                      # camada base e contratos
│   │   ├── __init__.py
│   │   ├── types.py               # contratos de dados          [Vitor]  feito
│   │   ├── settings.py            # ambiente + caminhos         [Vitor]  feito
│   │   ├── normalizers.py         # funções puras               [José]   feito
│   │   └── cli_utils.py           # resolução de imagens e CLI  [Vitor]  feito
│   ├── vision/                    # visão computacional OpenCV
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # OpenCV                      [Vitor]  feito
│   │   ├── qr_reader.py           # QR da AT                    [Vitor]  feito
│   │   └── evidence.py            # bbox -> recorte/destaque    [Vitor]  feito
│   ├── ocr/                       # motores de reconhecimento de texto
│   │   ├── __init__.py            # fábrica build_ocr_engine    [Vitor]  feito
│   │   ├── easyocr.py             # EasyOCR com linhas          [Vitor]  feito
│   │   └── tesseract.py           # backend Tesseract           [José]   feito
│   ├── extraction/                # extração estruturada e Q&A
│   │   ├── __init__.py
│   │   ├── llm_client.py          # local + cloud               [Vitor]  feito
│   │   ├── regex.py               # baseline heurístico         [Vitor]  feito
│   │   ├── llm.py                 # schema + prompt + parse     [Vitor]  feito
│   │   └── qa_assistant.py        # Q&A com evidência           [Vitor]  feito
│   └── evaluation/                # validação, scoring e benchmark
│       ├── __init__.py
│       ├── golden_set.py          # carregamento + validação    [José]   feito
│       ├── dataset_stats.py       # estatística do corpus       [José]   feito
│       ├── scoring.py             # comparação e agregação      [José]   feito
│       └── evaluate.py            # runner do benchmark         [José]   feito
├── tests/
│   ├── conftest.py                # fixtures partilhadas        [Vitor]  feito
│   ├── test_smoke.py              # arranque verde              [Vitor]  feito
│   ├── test_settings.py           # settings e .env             [Vitor]  feito
│   ├── test_preprocessing.py      # OpenCV e sonda OSD          [Vitor]  feito
│   ├── test_ocr_contract.py       # contrato do OcrEngine       [Vitor]  feito
│   ├── test_qr_reader.py          # QR sintético e parser       [Vitor]  feito
│   ├── test_evidence.py           # destaque e recorte          [Vitor]  feito
│   ├── test_normalizers.py        [José]   feito
│   ├── test_golden_set.py         [José]   feito
│   ├── test_scoring.py            [José]   feito
│   ├── test_ocr_tesseract.py      [José]   feito
│   ├── test_llm_client.py         [Vitor]  feito
│   ├── test_extractor_regex.py    [Vitor]  feito
│   ├── test_extractor_llm.py      [Vitor]  feito
│   ├── test_qa_assistant.py       [Vitor]  feito
│   ├── test_dataset_stats.py      [José]   feito
│   ├── test_evaluate.py           [José]   feito
│   └── test_app.py                [Vitor]  feito
├── run                            # lançador unificado Linux/macOS [Vitor] feito
├── run.bat                        # lançador unificado Windows     [Vitor] feito
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

`feito` significa integrado em `master` com testes a passar; `pendente`, não iniciado. A árvore
está atualizada a 2026-11-04. Os atalhos `run` e `run.bat` unificam os comandos via `poethepoet`,
assegurando execução idêntica em Windows, Linux e macOS.

### `.gitignore` (estado atual)

```text
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.ruff_cache/
.pytest_cache/
.coverage
.coverage.*
htmlcov/
models/*.gguf
models/*.bin
.env

# Ruído de sistema e de editor: a equipa trabalha em Linux, macOS e Windows.
.DS_Store
Thumbs.db
desktop.ini
*.log
*.swp
.idea/
.vscode/

# Artefactos do benchmark: as tabelas são versionadas, os dumps por imagem não.
results/*.json
```

As imagens de `data/receipts/` **são** versionadas, para que o corpus viaje com o repositório.
Se o risco de RGPD da secção 11 passar a pesar mais do que a conveniência, a linha
`data/receipts/*.jpg` pode ser acrescentada, mas nesse caso o corpus passa a ser distribuído
por outro meio e o README tem de explicar como.

---

## 3. Contratos de Dados (`src/core/types.py`)

Este ficheiro é a peça central da coordenação. É escrito pelo Vitor no Dia 0 e **não muda sem
aviso à equipa**. Todos programam contra ele. A fonte de verdade é o próprio ficheiro, que
documenta cada campo; esta secção resume o que ele contém.

Os tipos são todos `@dataclass(frozen=True, slots=True)` (`OcrEngine` é um `Protocol`), os
identificadores de linha começam em 1 e `status` é
`Literal["ENCONTRADO", "AUSENTE", "INVALIDO"]`. O módulo define também e exporta as constantes
partilhadas do projeto (`FIELD_NAMES`, `GROUPS`, `OCR_ENGINES`, `LLM_PROVIDERS`, `EXTRACTION_METHODS`,
`CONSUMIDOR_FINAL_NIF`, `EARLIEST_PLAUSIBLE_DATE`, `IMAGE_EXTENSIONS`, `STATUSES`, `ANSWER_STATUSES`,
`GROUND_TRUTH_COLUMNS`, `MATCH_RESULT_COLUMNS`, `METRIC_NAMES`, `ROTATION_CHOICES`), impõe tipos `Literal`
estritos (`OcrEngineName`, `MethodName`, `GroupName`, `FieldName`, `ProviderName`, `Status`, `AnswerStatus`,
`MetricName`, `RotationOption`) nos atributos das dataclasses e valida a ordenação geométrica não-negativa
(`x2 >= x1`, `y2 >= y1`) em `BBox.__post_init__`, além de exportar o validador partilhado `validate_image(image)`.
A implementação acrescentou três membros derivados, sem alterar nenhum campo do rascunho original:

| Tipo | Acrescento |
| :--- | :--- |
| `BBox` | propriedades `width` e `height`, além da validação em `__post_init__` |
| `OcrDocument` | `as_prompt_text()`, que numera as linhas como `[01] ...` para o prompt do LLM |
| `Extraction` | propriedade `fields`, que devolve os cinco campos por nome de coluna |

**Regra de ouro:** se um pacote de trabalho precisar de um campo novo nestes tipos, isso é uma
alteração de contrato e é feita pelo Vitor, não pelo autor do pacote.

---

## 4. Interface do Pipeline (o que cada fase consome e produz)

```text
imagem (np.ndarray BGR)
   │
   ├─ vision.preprocessing.preprocess(img, opts) -> PreprocessResult
   │     .image         imagem alinhada, recortada e com CLAHE
   │     .steps         dict[str, np.ndarray]  (bruta, rodada, recortada, filtrada)
   │     .rotation_deg  rotação aplicada
   │
   ├─ ocr.easyocr / ocr.tesseract.read(preprocessed.image) -> OcrDocument
   │
   ├─ vision.qr_reader.read(preprocessed.image) -> dict[str, str] | None
   │
   ├─ extraction.regex / extraction.llm.extract(OcrDocument, qr) -> Extraction
   │
   ├─ extraction.qa_assistant.ask(question, OcrDocument) -> Answer
   │
   └─ vision.evidence.draw_all_ocr_boxes(image, OcrDocument)               -> np.ndarray
      vision.evidence.highlight(image, OcrDocument, line_ids, alpha=0.25)  -> np.ndarray
      vision.evidence.crop(image, OcrDocument, line_ids, pad_px=16)        -> np.ndarray | None
```

`PreprocessResult` e as respetivas opções vivem em `preprocessing.py`, não em `types.py`,
porque só esse módulo as usa. `crop` devolve `None` quando nenhum dos `line_ids` existe no
documento, para que a interface possa simplesmente esconder o painel de recorte.

---

## 5. Pacotes de Trabalho

| ID | Título | Dono | Depende de | Tamanho | Ficheiros | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| WP-00 | Estrutura inicial do repositório | Vitor | — | 3 h | `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`, `tests/conftest.py`, `tests/test_smoke.py` | feito |
| WP-01 | Contratos de dados | Vitor | WP-00 | 2 h | `src/core/types.py` | feito |
| WP-02 | Normalizadores puros + testes | **José** | WP-01 | 4 a 6 h | `src/core/normalizers.py`, `tests/test_normalizers.py` | feito |
| WP-03 | Carregador e validador do golden set | **José** | WP-01, WP-02 | 4 a 6 h | `src/evaluation/golden_set.py`, `tests/test_golden_set.py` | feito |
| WP-04 | Pré-processamento OpenCV | Vitor | WP-01 | 8 h | `src/vision/preprocessing.py` | feito |
| WP-05 | Motor OCR EasyOCR | Vitor | WP-01, WP-04 | 6 h | `src/ocr/easyocr.py` | feito |
| WP-06 | Backend Tesseract | **José** | WP-05 | 3 a 4 h | `src/ocr/tesseract.py` | feito |
| WP-07 | Leitor de QR da AT | Vitor | WP-04 | 3 h | `src/vision/qr_reader.py` | feito |
| WP-08 | Cliente LLM unificado | Vitor | WP-01 | 6 h | `src/extraction/llm_client.py` | feito |
| WP-09 | Extrator por regex (baseline) | Vitor | WP-02, WP-05 | 5 h | `src/extraction/regex.py`, `tests/test_extractor_regex.py` | feito |
| WP-10 | Extrator por LLM | Vitor | WP-08, WP-09 | 6 h | `src/extraction/llm.py`, `tests/test_extractor_llm.py` | feito |
| WP-11 | Evidência visual | Vitor | WP-01 | 3 h | `src/vision/evidence.py` | feito |
| WP-12 | Assistente de Q&A com evidência | Vitor | WP-08, WP-11 | 5 h | `src/extraction/qa_assistant.py` | feito |
| WP-13 | Pontuação e agregação | **José** | WP-02, WP-03 | 5 a 7 h | `src/evaluation/scoring.py`, `tests/test_scoring.py` | feito |
| WP-14 | Estatística do corpus | **José** | WP-03 | 2 a 3 h | `src/evaluation/dataset_stats.py` | feito |
| WP-15 | Runner do benchmark | **José** | WP-13 | 4 a 5 h | `src/evaluation/evaluate.py`, `tests/test_evaluate.py` | feito |
| WP-16 | Aplicação Gradio | Vitor | WP-04 a WP-12 | 10 h | `src/app.py` | feito |
| WP-17 | Golden set: imagens + CSV | **Raquel** | — | 3 a 4 h | `data/receipts/*`, `data/ground_truth.csv` | feito |
| WP-18 | Mini-apresentação do golden set | **Raquel** | WP-17 | 3 h | `Docs/Presentation/` | pendente |

O caminho crítico inicial (WP-00 → WP-01 → WP-04 → WP-05 → WP-09 → WP-10 → WP-12 → WP-16),
o runner do benchmark oficial (WP-15) e o golden set (WP-17) encontram-se integralmente fechados e integrados em `master`.
O único pacote restante é a mini-apresentação (WP-18).

### 5.1 Estado das entregas e desvios face ao plano

Fechados e integrados em `master` a 2026-11-04: WP-00, WP-01, WP-02, WP-03, WP-04, WP-05, WP-06,
WP-07, WP-08, WP-09, WP-10, WP-11, WP-12, WP-13, WP-14, WP-15, WP-16 e WP-17. A suite tem 296 testes offline que correm
em cerca de 4,5 s, mais um teste `slow` com o EasyOCR real, deselecionado por omissão.

Desvios que valem registo, porque afetam os pacotes seguintes:

- **`src/core/settings.py` (fora da numeração).** Centraliza `os.getenv`, o carregamento do `.env` e
  a resolução de todos os caminhos, para cumprir a regra "nenhuma leitura de ambiente fora de
  `settings.py`". `load_settings(repo_root=...)` usa `dotenv_values` para construir um `Settings`
  isolado sem mutar `os.environ`; `get_settings()` é `lru_cache`ado e não deve ser usado em testes.
  Expõe configurações para `max_side`, `clahe_clip_limit`, `llama_port` e `llama_ngl`.
- **Organização modular em subpacotes.** Os módulos foram arrumados em 5 subpastas temáticas
  (`src/core/`, `src/vision/`, `src/ocr/`, `src/extraction/`, `src/evaluation/`), com `src/app.py`
  como entrada superior. Foi acrescentado `src/core/cli_utils.py` com funções partilhadas de
  carregamento e formatação de talões em CLI (`list_receipt_images`, `resolve_and_load_receipt`, `format_extraction_field`), e `src/ocr/__init__.py` com a fábrica `build_ocr_engine`.
- **Convenção de CLI.** Todos os módulos correm como `python -m src.<modulo>` e expõem
  `main(argv=None, settings_override=None)`. Os imports pesados (Pillow, EasyOCR, `settings`)
  ficam adiados para `main` ou para o construtor do motor. Isso mantém o import barato e é o
  que permite correr a suite em milissegundos.
- **Validação de imagens.** Todas as funções públicas que recebem uma imagem usam `validate_image(image)` de `src/core/types.py`, levantando
  `ValueError("Input image must not be empty or zero-sized.")` e devolvendo sempre uma cópia, sem
  mutar a entrada. Os testes de `evidence.py` verificam a não-destrutividade.
- **WP-04:** além da heurística de proporção, o recorte faz warp de perspetiva com suporte a
  `minAreaRect`, existe compensação contínua de inclinação (`auto_deskew=False` por omissão), e a
  verificação de rotação a 180° combina uma heurística de centróide vertical do QR com a sonda
  Tesseract OSD (`--psm 0`, timeout de 5 s, com encaminhamento de `tesseract_cmd`).
- **WP-05:** `EasyOCREngine` aceita `reader=` para injeção em testes e agrupa fragmentos na mesma
  linha horizontal antes de atribuir identificadores sequenciais de 1 a N.
- **WP-07:** a descodificação itera por todos os códigos detetados (`_decode_all_codes`), impedindo
  que códigos promocionais mascarem o QR fiscal da AT. O parser mantém as chaves cruas da AT e
  acrescenta as alias usadas pelo resto do pipeline.
- **WP-11:** a API ficou `draw_all_ocr_boxes`, `highlight` e `crop`, com `crop` a devolver
  `None` quando nenhum `line_id` existe. `highlight` desenha as linhas não citadas em cinza
  esbatido, para que a evidência ativa salte à vista.
- **WP-02:** os normalizadores são puros e não leem ficheiros, ambiente nem desenham nada.
  `parse_data` utiliza `re.finditer` para testar todos os candidatos da linha, evitando saídas
  prematuras provocadas por números de referência; `validar_data` verifica `2000-01-01 <= parsed <= today`;
  `parse_nif` normaliza prefixos `PT` distorcidos por OCR, recupera sequências de 10 dígitos iniciadas por `7` e repara barras (`/`) para `7` ou `1` quando validadas pelo dígito de controlo;
  `normalizar_entidade` remove sufixos societários (`ENTITY_SUFFIXES`: `lda`, `sa`, `unipessoal`, `unip`, `sociedade`, `limitada`, `s`) preservando pelo menos um token; e `valores_equivalentes` valida `campo` antes de qualquer verificação de `None`.
- **WP-06:** `TesseractEngine` agrega caixas de palavra por `(block_num, par_num, line_num)`,
  descarta `conf < 0`, expõe `image_to_data_fn` para injeção de testes e restaura o
  `pytesseract.pytesseract.tesseract_cmd` global no fim.
- **WP-08:** `llm_client` expõe `OpenAICompatibleClient` (que serve o `llama-server` local),
  `GeminiClient` (com sanitização recursiva de `additionalProperties` via `_sanitize_gemini_schema`),
  `FailoverLLMClient`, e utilitários partilhados (`build_object_schema`, `parse_json_object`, `coerce_line_id`). O suporte ao provedor direto `llm_openai` foi removido para focar a
  comparação em Local vs. Google Gemini.
- **WP-09:** ver a secção 6, WP-09, para as regras do baseline e as três armadilhas que o
  pacote fecha (sentinela do consumidor final, NIF duplicado e total com quantidade à frente).
  O padrão `_TOTAL_LINE` suporta qualificadores (`A PAGAR`, `LIQUIDO`, `GERAL`, `FINAL`) e
  prefixos de moeda.
- **WP-10:** `extract` aceita um `client` injetável e, sem ele, constrói o provedor a partir do
  ambiente. Reconcilia `nif_comerciante` e `nif_cliente` contra a extração regex via `_reconcile_nifs`, recorre ao fallback regex para `data` e `total` quando o LLM devolve `AUSENTE`, e o `_locate_line` para o total pesquisa em ordem inversa (`reversed(lines)`) priorizando
  linhas com palavras-chave de total para não ancorar em linhas de artigos.
- **WP-15:** `avaliar_registo` inclui verificação pós-OCR de orientação por palavras-chave fiscais, rodando a imagem 180° e repetindo o OCR quando a primeira passagem encontra `<= 2` palavras-chave e a imagem invertida produz `>= 3` palavras-chave (e pelo menos mais 3 do que a orientação inicial).
- **Arranque do `llama-server` (`./run serve` / `run serve`).** O modelo por omissão é
  `models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf`, com porta 8080 e `NGL=0`.
- **Validação do golden set (`./run check-dataset`).** Atalho que executa `src.evaluation.golden_set` e
  `src.evaluation.dataset_stats`.

Estado atual do golden set e do benchmark: o corpus conta com 27 imagens em `data/receipts/` e
`data/ground_truth.csv` contém os 27 registos correspondentes com as designações comerciais de
`entidade` alinhadas aos cabeçalhos impressos nos talões (WP-17 concluído), passando a validação de
`./run check-dataset` a 100 % e gerando os relatórios oficiais em `results/metrics.csv` e
`results/metrics.md`.

---

## 6. Especificação Detalhada dos Pacotes de dados e métricas

Regras gerais, válidas para todos os pacotes abaixo:

- Trabalhar sempre num branch próprio: `feat/wp-02-normalizers`.
- Um branch = um pacote de trabalho. No máximo ~200 linhas de diff.
- Não alterar ficheiros fora da lista do pacote.
- Correr antes de abrir o merge request:

  ```bash
  uv run ruff format --line-length 120 src tests
  uv run ruff check src tests
  uv run pyright src tests
  uv run pytest -q
  ```

- Docstrings no estilo Google, em inglês, como no `nlpg-6ed-group-4`.

---

### WP-02 — Normalizadores puros (`src/core/normalizers.py`)

**Assinaturas a implementar:**

```python
def limpar_texto(raw: str) -> str:
    """Collapse whitespace, drop control characters and trim the ends."""


def parse_moeda(raw: str) -> Decimal | None:
    """Convert '16,10 EUR', '11.57', '5,86€', ' 1 234,56 ' into Decimal.

    Returns None when no interpretable number is present.
    Treats '.' as a thousands separator when followed by exactly three digits.
    """


def parse_data(raw: str) -> date | None:
    """Recognise '2026-10-24', '26-10-27', '26-10-27 13:45:42', '20261027'.

    Assumes the 2000s century for two-digit years.
    """


def validar_data(parsed: date) -> bool:
    """Validate whether a parsed receipt date falls between 2000-01-01 and today inclusive."""


def parse_nif(raw: str) -> str | None:
    """Extract nine digits from 'NIF: 518396894', 'NIF 518 396 894', '518396894'."""


def validar_nif(nif: str) -> bool:
    """Validate the modulo-11 check digit and the permitted first digit."""


def normalizar_entidade(raw: str) -> str:
    """Canonical form for comparison: lowercase, no accents, no punctuation,
    no corporate suffixes (lda, sa, unipessoal, unip, sociedade, limitada, s), whitespace collapsed."""


def valores_equivalentes(campo: FieldName, esperado: str | None, obtido: str | None) -> bool:
    """Compare two already-converted values using the per-field rule.

    Rules: 'total' compares Decimals; 'data' compares dates; 'nif_*' compares
    nine-digit strings; 'entidade' compares canonical forms, accepting either
    form as a token-boundary substring of the other, or a SequenceMatcher ratio
    of at least ENTITY_SIMILARITY_THRESHOLD (0.72) on the whole forms or on a
    sliding window of the obtained tokens.
    Two Nones are equivalent. An unknown field raises ValueError before any
    absence check.
    """
```

**Tabela de aceitação (a transformar em `pytest.mark.parametrize`):**

| Entrada | Função | Saída esperada |
| :--- | :--- | :--- |
| `"16,10 EUR"` | `parse_moeda` | `Decimal("16.10")` |
| `"11.57"` | `parse_moeda` | `Decimal("11.57")` |
| `"1 234,56"` | `parse_moeda` | `Decimal("1234.56")` |
| `"TOTAL"` | `parse_moeda` | `None` |
| `"26-10-27 13:45:42"` | `parse_data` | `date(2026, 10, 27)` |
| `"20261027"` | `parse_data` | `date(2026, 10, 27)` |
| `"31-02-26"` | `parse_data` | `None` |
| `"NIF: 518 396 894"` | `parse_nif` | `"518396894"` |
| `"518396894"` | `validar_nif` | `True` |
| `"518396895"` | `validar_nif` | `False` |
| `"O DAS BIFANAS, LDA."` | `normalizar_entidade` | `"o das bifanas"` |
| `("total", "16,10", "16.1")` | `valores_equivalentes` | `True` |
| `("entidade", None, None)` | `valores_equivalentes` | `True` |
| `("entidade", "Pingo Doce", "Continente")` | `valores_equivalentes` | `False` |

**Critério de aceitação:** todas as linhas acima passam, mais pelo menos seis casos
adicionais inventados pelo autor e documentados no teste. Cobertura de `normalizers.py` acima
de 95 %.

**O que NÃO fazer:** não ler ficheiros, não usar `print`, não chamar `open()`, não aceder a
variáveis de ambiente.

---

### WP-03 — Carregador e validador do golden set (`src/evaluation/golden_set.py`)


**Contrato de entrada (`data/ground_truth.csv`), colunas obrigatórias:**

```text
filename,grupo,entidade,nif_comerciante,nif_cliente,data,total
```

- `filename` — nome do ficheiro dentro de `data/receipts/`, com extensão.
- `grupo` — `direito` ou `amarrotado`.
- `entidade` — designação como aparece no talão; vazio se ilegível.
- `nif_comerciante` — nove dígitos; vazio apenas se ausente no documento.
- `nif_cliente` — nove dígitos; vazio quando o talão não tem contribuinte.
- `data` — `AAAA-MM-DD`.
- `total` — decimal com ponto (`16.10`), sem símbolo de moeda.

**Assinaturas:**

```python
def carregar_ground_truth(path: Path, receipts_dir: Path | None = None) -> list[GroundTruthRecord]:
    """Read and validate the CSV. Raise GoldenSetError with every failure found."""


def validar_ground_truth(path: Path, receipts_dir: Path) -> list[str]:
    """Return the list of problems found; an empty list means valid."""


@dataclass(frozen=True, slots=True)
class GoldenSetError(Exception):
    """Aggregate every validation error into one readable message."""
```

**Regras de validação a implementar (todas obrigatórias):**

1. Cabeçalho exatamente com as sete colunas esperadas, pela ordem indicada.
2. `filename` não vazio, sem duplicados e existente em `receipts_dir`.
3. `grupo` em `{"direito", "amarrotado"}`.
4. `nif_comerciante` e `nif_cliente`, quando preenchidos, têm nove dígitos, passam
   `validar_nif` e rejeitam o código sentinela de consumidor final (`999999990`).
5. `data`, quando preenchida, é interpretável, não está no futuro e não é anterior a `2000-01-01`.
6. `total`, quando preenchido, é interpretável e maior que zero.
7. Nenhuma linha totalmente vazia; nenhuma coluna extra.

**Critério de aceitação:** um teste que constrói CSVs inválidos em `tmp_path` (ficheiro
inexistente, NIF inválido, data no futuro, duplicado, coluna em falta) e verifica que
`validar_ground_truth` devolve o erro correspondente. Um teste contra o CSV real do repositório
que espera uma lista vazia.

---

### WP-06 — Backend Tesseract (`src/ocr/tesseract.py`)

**Assinatura:**

```python
class TesseractEngine:
    """Tesseract-based OCR engine, aggregating word boxes into lines."""

    name = "tesseract"

    def __init__(
        self,
        lang: str = "por",
        psm: int = 6,
        tesseract_cmd: Path | str | None = None,
        image_to_data_fn: Any | None = None,
    ) -> None: ...

    def read(self, image: np.ndarray) -> OcrDocument: ...
```

**Detalhes de implementação:**

- Usar `pytesseract.image_to_data(image, lang=self.lang, config=f"--psm {self.psm}",
  output_type=pytesseract.Output.DICT)`.
- Agrupar por `(block_num, par_num, line_num)`.
- Descartar `conf < 0` (o Tesseract usa `-1` para tokens sem confiança).
- A caixa da linha é o invólucro de todas as caixas de palavra dessa linha.
- O `id` é atribuído sequencialmente a partir de 1, por ordem de leitura (de cima para baixo,
  da esquerda para a direita).
- O texto da linha é a junção das palavras com um único espaço.

**Critério de aceitação:** correr sobre `data/receipts/01_*.jpg` e obter mais de 10 linhas,
todas com `bbox` dentro dos limites da imagem e `id` sequencial sem falhas.

---

### WP-09 — Extrator por regex (`src/extraction/regex.py`)

Este pacote não tinha especificação detalhada na primeira versão do direito, só a linha da
tabela da secção 5. As regras abaixo são as que ficaram implementadas e testadas, e passam a
ser o contrato do baseline.

**Assinatura:**

```python
def extract(doc: OcrDocument, qr: dict[str, str] | None = None) -> Extraction:
    """Extraction com method='regex', elapsed_ms medido e raw_response=None."""


def main(argv: Sequence[str] | None = None, settings_override: Settings | None = None) -> None:
    """Pré-visualização: --file, --engine {easyocr,tesseract}, --no-preprocess."""
```

**Regras de extração, por campo:**

- `entidade` — primeira linha que pareça um nome: pelo menos 3 letras, mais de metade dos
  caracteres visíveis alfabéticos, sem palavras de cabeçalho/rodapé (`fatura`, `recibo`,
  `contribuinte`, `nif`, `atcud`, `cliente`, `total`, `iva`, `tel`, `www`, ...) e com duas
  palavras ou uma palavra de 5 letras ou mais. Vale a primeira linha plausível, não a mais
  plausível: o emitente é impresso antes da morada.
- `nif_comerciante` — primeiro NIF de uma linha com rótulo de emitente (`NIF`, `NIPC`,
  `CONTRIBUINTE`, `EMITENTE`); na falta dele, primeiro NIF sem rótulo dentro da faixa de topo
  (40 % superiores da imagem).
- `nif_cliente` — **último** NIF de uma linha com rótulo de cliente (`CLIENTE`, `ADQUIRENTE`,
  `CONSUMIDOR`). O último é o que faz funcionar uma linha única do tipo
  `NIF: 500100144 NIF CLIENTE: 123456789`.
- `data` — primeira data plausível em ordem de leitura, em ISO 8601. Datas posteriores a hoje
  são descartadas: como as datas com hífen são lidas como `YY-MM-DD`, uma leitura errada de um
  talão `DD-MM-YY` cai na década seguinte, e uma data de emissão no futuro é impossível.
- `total` — **última** linha que case com rótulo de total seguido do valor no fim da linha
  (`TOTAL`, `TOTAL A PAGAR`, `VALOR TOTAL`), com separador de milhares suportado. O valor
  fica com duas casas decimais e ponto (`16.10`). Totais zero ou negativos são `INVALIDO`.

**Três armadilhas que este pacote fecha, e que os pacotes seguintes herdam:**

1. `999999990` é o código de consumidor final da AT e **passa** o dígito de controlo do NIF.
   É rejeitado por uma constante explícita (`CONSUMIDOR_FINAL_NIF`), não por `validar_nif`, e
   resulta em `AUSENTE`. Sem isto o sistema reportaria um NIF de cliente inventado.
2. Um NIF de cliente igual ao do comerciante é descartado: um talão identifica sempre o
   emitente, por isso quando só existe um número ele pertence ao comerciante.
3. `parse_moeda` devolve o **primeiro** número da string, pelo que aplicá-la à linha inteira
   leria `TOTAL 2 UN 16,10` como `2`. O total é lido com o padrão próprio do módulo, que exige
   duas casas decimais e o valor no fim da linha, e uma linha com ruído entre o rótulo e o
   valor é descartada em vez de mal lida.

**Regra de ouro:** o payload do QR é aceite na assinatura por paridade com o extrator por LLM
e com a aplicação, mas **nunca** preenche um campo. O baseline tem de ser medido só sobre o
texto do OCR, e os talões de terminal de pagamento não têm QR.

**Critério de aceitação:** os testes em `tests/test_extractor_regex.py` cobrem os cinco campos,
as três armadilhas acima, ausência de todos os campos num documento vazio e a pré-visualização
de linha de comando com um motor injetado. Nenhum teste descodifica uma imagem real.

---

### WP-13 — Pontuação e agregação (`src/evaluation/scoring.py`)

**Assinaturas:**

```python
def pontuar_registo(
    verdade: GroundTruthRecord,
    extracao: Extraction,
) -> list[MatchResult]:
    """Compare the five fields and return one MatchResult per field."""


def agregar(resultados: list[MatchResult]) -> dict[str, dict[str, float]]:
    """Aggregate by field and by group.

    Structure: {"por_campo": {...}, "por_grupo": {...}, "global": {...}}
    """


def escrever_tabelas(
    resultados: Sequence[MatchResult],
    destino: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write results/metrics.csv and results/metrics.md (table for the report)."""
```

**Definição formal das métricas (usar exatamente esta):**

Para cada `MatchResult`:

- `acertou = valores_equivalentes(campo, esperado, obtido)`
- `falso_positivo = esperado is None and obtido is not None`
- `omissao = esperado is not None and obtido is None`

Agregações por campo e por grupo:

- `taxa_acerto = acertou / n`
- `taxa_falso_positivo = falso_positivo / n_esperados_ausentes`
- `taxa_omissao = omissao / n_esperados_presentes`

Deixar explícito no docstring que `taxa_acerto` conta os casos ausentes como acertados quando o
sistema também devolve `AUSENTE`. É isto que separa "leu bem" de "inventou".

**Critério de aceitação:** testes construídos à mão que exercitam os três contadores, incluindo
o caso em que o sistema inventa um NIF de cliente num talão que não tem contribuinte.

---

### WP-14 — Estatística do corpus (`src/evaluation/dataset_stats.py`)

**Assinatura:**

```python
def resumo(registos: list[GroundTruthRecord]) -> dict[str, object]:
    """Counts by group, by merchant and by presence of each field."""


def imprimir_resumo(registos: list[GroundTruthRecord]) -> None:
    """Format the summary as a text table for the terminal."""
```

Saída esperada:

```text
Grupo          N   entidade  nif_com  nif_cli   data  total
direito         12        12        12       7      12     12
amarrotado     8         7         8       5       8      8
```

---

### WP-15 — Runner do benchmark (`src/evaluation/evaluate.py`, `tests/test_evaluate.py`)

**Dono:** **José** | **Branch:** `feat/wp-15-evaluate` | **Depende de:** WP-03, WP-09, WP-10, WP-13

**Objetivo:**
Orquestrar a avaliação fim a fim do pipeline sobre todos os documentos do golden set
(`data/ground_truth.csv`), pontuando cada extração contra a verdade de referência com
`pontuar_registo` e gravando os artefactos finais em `results/metrics.csv` e `results/metrics.md`
(que são lidos diretamente pelo separador 3 da aplicação Gradio e citados no relatório).

**Convenções obrigatórias do repositório (verificar antes de abrir MR):**
1. **Docstring de topo com referência:** incluir `Ref: Docs/implementation_plan.md, section 6, WP-15.`
2. **Imports pesados adiados:** no topo do ficheiro importar apenas tipos leves (`src.core.types`, `pathlib`, `collections.abc`, `argparse`). Adiar `cv2`, `EasyOCREngine`, `TesseractEngine`, `build_client`, `get_settings` para dentro de `main()` ou das funções quando os objetos injetáveis forem `None`.
3. **Sem `print` em funções de biblioteca:** `avaliar_corpus` não faz `print`; toda a saída de consola vive em `main()`.
4. **Sem leitura direta de variáveis de ambiente:** usar exclusivamente `settings_override or get_settings()`.

**Assinaturas a implementar em `src/evaluation/evaluate.py`:**

```python
MethodName = Literal["regex", "llm_local", "llm_gemini"]


def avaliar_registo(
    registo: GroundTruthRecord,
    receipts_dir: Path,
    ocr_engine: OcrEngine,
    *,
    method: MethodName = "regex",
    llm_client: LLMClient | None = None,
    preprocess_opts: PreprocessOptions | None = None,
    settings: Settings | None = None,
) -> tuple[Extraction, list[MatchResult]]:
    """Run preprocessing, OCR, QR reading, extraction, and scoring for a single receipt.

    Args:
        registo: Ground-truth reference record for one receipt.
        receipts_dir: Directory containing the receipt image files.
        ocr_engine: Initialised OCR engine instance (EasyOCR or Tesseract).
        method: Extraction method ('regex', 'llm_local', or 'llm_gemini').
        llm_client: Optional injected LLM client (used when method != 'regex').
        preprocess_opts: Optional OpenCV preprocessing options. When None, default options apply.
        settings: Optional Settings instance for configuring preprocessing or client fallback.

    Returns:
        Tuple of (Extraction, list of 5 MatchResult objects).

    Raises:
        FileNotFoundError: If the image file ``receipts_dir / registo.filename`` does not exist.
        ValueError: If ``cv2.imread`` fails to decode the image.
    """


def avaliar_corpus(
    registos: Sequence[GroundTruthRecord],
    receipts_dir: Path,
    ocr_engine: OcrEngine,
    *,
    method: MethodName = "regex",
    llm_client: LLMClient | None = None,
    preprocess_opts: PreprocessOptions | None = None,
    destino_resultados: Path | None = None,
    settings: Settings | None = None,
    progress_callback: Callable[[int, int, GroundTruthRecord], None] | None = None,
) -> list[MatchResult]:
    """Evaluate a sequence of ground-truth records and optionally write report tables.

    Args:
        registos: Validated ground-truth records to evaluate.
        receipts_dir: Directory containing the receipt images.
        ocr_engine: Initialised OCR engine instance, reused across all receipts.
        method: Extraction method to benchmark.
        llm_client: Optional injected LLM client.
        preprocess_opts: Optional OpenCV preprocessing options.
        destino_resultados: When provided, calls ``escrever_tabelas(resultados, destino_resultados)``
            to generate ``metrics.csv`` and ``metrics.md``.
        settings: Optional Settings instance forwarded to record evaluation.
        progress_callback: Optional callback invoked before processing each receipt.

    Returns:
        Combined list of ``MatchResult`` objects across all evaluated receipts (5 per receipt).
    """


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> None:
    """CLI entry point to run the benchmark and write metrics to results/."""
```

**Fluxo passo a passo dentro de `avaliar_registo`:**

1. Construir o caminho da imagem: `image_path = receipts_dir / registo.filename`. Se não existir (`not image_path.is_file()`), levantar `FileNotFoundError(f"Receipt image not found: {image_path}")`.
2. Ler a imagem com `cv2.imread(str(image_path))`. Se devolver `None`, levantar `ValueError(f"Unable to decode receipt image: {image_path}")`.
3. Pré-processar a imagem: `prep = preprocess(raw_image, preprocess_opts or PreprocessOptions())`.
4. Correr o OCR sobre a imagem pré-processada: `doc = ocr_engine.read(prep.image)`. Se a leitura inicial produzir `<= 2` palavras-chave fiscais (`total`, `fatura`, `nif`, `iva`, etc.) e a imagem rodada a 180° produzir `>= 3` palavras-chave (e pelo menos mais 3 do que a orientação inicial), adotar a imagem invertida e o respetivo `OcrDocument`.
5. Ler o QR Code (para paridade de assinatura): `qr_data = read_qr(prep.image, raw_image=raw_image)`.
6. Extrair os 5 campos estruturados:
   - Se `method == "regex"`: chamar `extract_regex(doc, qr=qr_data)`.
   - Caso contrário (`"llm_local"`, `"llm_gemini"`): se `llm_client` for `None`, construir com `build_client(method)`; chamar `extract_llm(doc, qr=qr_data, client=llm_client)`.
7. Pontuar contra a verdade de referência: `matches = pontuar_registo(registo, extraction)`.
8. Devolver `(extraction, matches)`.

**Fluxo passo a passo dentro de `avaliar_corpus`:**

1. Se `method != "regex"` e `llm_client is None`, instanciar um único `llm_client = build_client(method)` antes do ciclo para o reutilizar em todos os recibos.
2. Iterar por cada `registo` em `registos`, acumular os 5 `MatchResult` devolvidos por `avaliar_registo` numa lista `todos_resultados: list[MatchResult]`.
3. Se `destino_resultados is not None`, chamar `escrever_tabelas(todos_resultados, destino_resultados)` (função já implementada pelo José em `src/scoring.py`).
4. Devolver `todos_resultados`.

**Argumentos de linha de comando em `main()`:**

Usar `argparse.ArgumentParser` com as seguintes flags:
- `--csv` (`type=Path`, `default=settings.ground_truth_path`): caminho para o ficheiro `ground_truth.csv`.
- `--receipts-dir` (`type=Path`, `default=settings.receipts_dir`): pasta das imagens.
- `--output-dir` (`type=Path`, `default=settings.results_dir`): diretoria onde serão escritos `metrics.csv` e `metrics.md`.
- `--engine` (`choices=("easyocr", "tesseract")`, `default=settings.ocr_engine`): motor OCR a utilizar.
- `--method` (`choices=("regex", "llm_local", "llm_gemini")`, `default="regex"`): método de extração a avaliar.
- `--no-preprocess` (`action="store_true"`): quando ativo, passa `PreprocessOptions(rotation_deg=0, auto_crop=False, remove_shadows=False, clahe_clip_limit=0.0, auto_orient_osd=False)`.

Comportamento de `main()`:
1. Carregar os registos com `carregar_ground_truth(args.csv)`. Se levantar `GoldenSetError`, imprimir os erros e sair sem exceção não tratada (tal como em `dataset_stats.py`).
2. Instanciar o motor OCR escolhido uma única vez (`TesseractEngine(tesseract_cmd=settings.tesseract_cmd)` ou `EasyOCREngine(langs=settings.easyocr_langs)`).
3. Chamar `avaliar_corpus(..., destino_resultados=args.output_dir)`.
4. Calcular o resumo agregado com `agregar(resultados)` e imprimir no terminal a taxa de acerto global (`taxa_acerto`), a taxa de falsos positivos (`taxa_falso_positivo`), a taxa de omissão (`taxa_omissao`) e o caminho onde `metrics.csv` e `metrics.md` foram gravados.

**Casos de teste obrigatórios em `tests/test_evaluate.py` (100 % offline):**

1. **`test_avaliar_registo_regex_pipeline_completo`:**
   - Criar uma imagem sintética em `tmp_path` (ex. `cv2.imwrite` de uma matriz branca `np.full((80, 80, 3), 255, dtype=np.uint8)`).
   - Criar um `FakeOcrEngine` que devolve um `OcrDocument` fixo com linhas conhecidas (`PINGO DOCE`, `NIF: 500100144`, `DATA: 2026-10-24`, `TOTAL: 16,10`).
   - Passar um `GroundTruthRecord` correspondente e verificar que `avaliar_registo` devolve `Extraction` com `method="regex"` e 5 `MatchResult` com `acertou=True`.
2. **`test_avaliar_registo_llm_com_cliente_injetado`:**
   - Usar o mesmo `FakeOcrEngine` e injetar um `FakeLLMClient` que devolve um JSON válido.
   - Verificar que `avaliar_registo(..., method="llm_local", llm_client=fake_client)` usa o extrator por LLM e pontua os 5 campos.
3. **`test_avaliar_registo_ficheiro_inexistente_levanta_erro`:**
   - Verificar que um `GroundTruthRecord` cujo `filename` não existe em `tmp_path` levanta `FileNotFoundError`.
4. **`test_avaliar_corpus_escreve_metrics_csv_e_md`:**
   - Passar dois `GroundTruthRecord` (um `"direito"` e um `"amarrotado"`) e `destino_resultados=tmp_path / "results"`.
   - Verificar que `avaliar_corpus` devolve 10 `MatchResult` e cria `metrics.csv` e `metrics.md` dentro de `tmp_path / "results"`.
5. **`test_main_cli_executa_offline_com_doubles`:**
   - Criar um CSV válido de 1 linha e 1 imagem sintética em `tmp_path`.
   - Usar `monkeypatch` para substituir `EasyOCREngine` pelo motor falso.
   - Chamar `main(argv=["--csv", str(csv_path), "--receipts-dir", str(receipts_dir), "--output-dir", str(out_dir), "--no-preprocess"], settings_override=settings)`.
   - Verificar com `capsys` que imprime as métricas finais e que `out_dir / "metrics.md"` foi criado.

**Critério de aceitação:**
Todos os testes em `tests/test_evaluate.py` passam offline em menos de 1 segundo, sem carregar o EasyOCR real, sem chamar o `tesseract` e sem aceder à rede. `ruff format`, `ruff check` e `pyright` passam com zero erros.

---

## 7. Especificação dos Pacotes da pipeline

**WP-04 `src/vision/preprocessing.py`** (implementado)

- `resize_max_side(img, 1280)` antes de tudo.
- Orientação em dois estágios: heurística por proporção (um talão vertical fotografado na
  horizontal roda 90° no sentido horário) e, já depois do filtro, uma sonda Tesseract OSD
  (`--psm 0`, timeout de 5 s). A sonda corrige fotos de cabeça para baixo que a proporção não
  deteta e nunca quebra o pipeline: qualquer erro ou ausência do binário devolve 0.
- Recorte: escala de cinzentos → desfoque gaussiano → Canny → dilatação → contornos → maior
  quadrilátero com área superior a 20 % da imagem, seguido de warp de perspetiva; recurso
  alternativo à imagem inteira.
- Normalização de iluminação: `gray / close(gray, kernel=25x25) * 255`.
- CLAHE com `clipLimit=2.0`, `tileGridSize=(8, 8)`; ambos configuráveis em `PreprocessOptions`,
  a par de `auto_crop`, `remove_shadows` e `auto_orient_osd`.
- `steps` devolve as imagens intermédias `"bruta"`, `"rodada"`, `"recortada"` e `"filtrada"`
  para o segundo separador.

**WP-05 `src/ocr/easyocr.py`** (implementado)

- `EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True, reader=None)`; o `reader` é o
  ponto de injeção usado pelos testes, que assim nunca carregam pesos.
- Pré-aquecimento com uma imagem sintética no arranque, para que a primeira inferência real
  não pague o custo de carregar os pesos.
- Mapear `readtext(detail=1)` para `OcrLine` com `id` sequencial: caixas clampadas à imagem,
  texto vazio e caixas de área nula descartados, e ordenação por `(y1, x1)` antes de numerar.

**WP-07 `src/vision/qr_reader.py`** (implementado)

- `cv2.QRCodeDetector().detectAndDecode` e, em alternativa, `pyzbar.decode`. O `pyzbar` é
  tentado primeiro mas é opcional: sem a biblioteca de sistema `zbar`, o import falha em
  silêncio e o OpenCV assume.
- Cinco estágios de descodificação: imagem direta, cinzentos, limiar adaptativo, procura de
  blobs quadrados com reparação das riscas verticais deixadas por cabeças térmicas avariadas,
  e nova tentativa sobre a imagem bruta na resolução original.
- Interpretar o conteúdo da AT: pares `CHAVE:VALOR` separados por `*`. O dicionário devolvido
  mantém as chaves cruas (`A`, `B`, `F`, `H`, `O`, ...) e acrescenta as alias
  `nif_comerciante`, `nif_cliente`, `data`, `total`, `atcud` e `tipo_doc`.
- `B = 999999990` é o código de consumidor final e mapeia para `nif_cliente` vazio.
- Uso: validação cruzada e coluna extra na interface. **Nunca** substitui o OCR, porque os
  talões de terminal de pagamento não têm QR.

**WP-08 `src/extraction/llm_client.py`** (implementado)

```python
class LLMClient(Protocol):
    provider: str
    last_effective_provider: str

    def complete(
        self,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        image: np.ndarray | None = None,
    ) -> str: ...


def build_client(
    provider: str = "llm_local",
    settings: Settings | None = None,
    primary_client: Any | None = None,
    fallback_client: Any | None = None,
) -> LLMClient: ...
```

- `llama-server` usa `OpenAICompatibleClient` (`openai.OpenAI(base_url=...)`) para inferência local com Qwen2.5.
- O Gemini usa `GeminiClient` (`google-genai`); o helper `_sanitize_gemini_schema` remove recursivamente
  `additionalProperties` para evitar erros HTTP 400 da API da Google.
- Failover: em erro de rede, erro de quota ou timeout, recua para o cliente local e marca o
  `last_effective_provider` na resposta.

**WP-10 `src/extraction/llm.py`** (implementado)

**Assinatura:**

```python
def extract(
    doc: OcrDocument,
    qr: dict[str, str] | None = None,
    client: LLMClient | None = None,
) -> Extraction:
    """Extraction com method=<provedor que respondeu>, elapsed_ms medido e raw_response."""


def main(argv: Sequence[str] | None = None, settings_override: Settings | None = None) -> None:
    """Pré-visualização: --file, --engine, --provider, --no-preprocess, --show-raw."""
```

- Prompt de sistema em português, com o texto indexado linha a linha (`[01] ...`) como mensagem
  do utilizador. O `documento.as_prompt_text()` é a única fonte de texto: o QR é aceite na
  assinatura e ignorado, tal como no baseline.
- Schema direito, sem uniões de tipo: os cinco campos são `string`, a ausência é `""`, e há dois
  objetos fechados, `presente` (booleanos) e `linhas` (inteiros). **Todos** os objetos levam
  `additionalProperties: false`, garantindo gramática estrita no `llama-server` (e sendo sanitizado
  automaticamente no cliente Gemini).
- **Ponto confirmado:** o schema em `response_format` **não é injetado no prompt** pelo
  `llama-server`. A estrutura está descrita em texto no próprio prompt, caso contrário o modelo
  produz JSON válido com as chaves erradas.
- Recuperação do JSON: tolera bloco de código cercado e uma frase de prosa antes ou depois do
  objeto. Sem objeto descodificável, ou com uma chave obrigatória em falta, levanta
  `LLMResponseError`. Chaves desconhecidas são registadas em `logger.warning` e ignoradas.
- `presente` é autoritativo por omissão: quando o modelo devolve texto mas marca `presente` a `false`, o valor só é aceite se passar a validação semântica do campo e puder ser fisicamente ancorado numa linha do OCR; caso contrário resulta em `AUSENTE`, impedindo falsos positivos num talão sem NIF de cliente.
- Pós-processamento, sempre a desconfiar do modelo: `parse_nif` + `validar_nif` para os NIF (com
  o sentinela do consumidor final a resultar em `AUSENTE`), `parse_data` + `validar_data` para a data (com
  rejeição de datas futuras ou anteriores a `2000-01-01`), `parse_moeda` para o total (com rejeição de zero ou negativo). O
  que não sobrevive fica `INVALIDO` e conserva o texto, para a pontuação contar como erro e não
  como omissão. Adicionalmente, `_reconcile_nifs` desambigua `nif_comerciante` e `nif_cliente` com recurso à extração regex sobre o OCR (corrigindo trocas, truncagens ou códigos postais), e `data` / `total` marcados como `AUSENTE` pelo LLM recorrem ao fallback regex quando este encontra um valor válido.
- O prompt proíbe expressamente somas e cálculos (regra 2 do sistema). Sem isto, o modelo local
  real somou duas linhas de artigos (0,80 + 2,40 = 3,20) e marcou o total como presente num talão
  que não tinha total impresso.
- Ancoragem determinística (`_locate_line`): um `line_id` inventado ou devolvido como `0` é
  procurado diretamente nas linhas do OCR através do normalizador próprio de cada campo. Para o total,
  a pesquisa é realizada em ordem inversa (`reversed(lines)`) priorizando linhas com palavras-chave
  (`total`, `pagar`, `liquido`, `final`, `soma`) para evitar ancoragens falsas em preços de artigos.
- Um identificador que não exista no documento é descartado e re-ancorado; se o valor não
  existir de todo no texto, fica sem linha.
- Documento sem linhas não gera pedido ao modelo: devolve cinco `AUSENTE` e não custa uma
  chamada nem é pontuado como falha do modelo.

**Limitação conhecida:** `parse_data` não reconhece meses escritos por extenso, pelo que um
modelo que responda `27 de outubro de 2026` produz `INVALIDO`. O prompt pede `AAAA-MM-DD`, e
alargar o normalizador é uma alteração do WP-02, não deste pacote. O comportamento está fixado
por teste em vez de ficar implícito.

**Critério de aceitação:** os testes em `tests/test_extractor_llm.py` (47 testes) injetam sempre
um cliente falso; nenhum pedido sai do processo. Cobrem os cinco campos, a contradição
`presente`/valor, o sentinela do consumidor final, NIF inválido, data impossível e data futura,
total zero, JSON cercado com prosa, resposta sem JSON, resposta com chave em falta, as regras do
prompt (proibição de cálculos e exigência de linhas), substituição de `line_id` inventado ou
zero pela linha real localizada, valor inventado sem linha atribuída, documento vazio e as três
saídas de erro da linha de comando (modelo indisponível, resposta inutilizável, ficheiro
ausente).

**WP-11 `src/vision/evidence.py`** (implementado)

- `draw_all_ocr_boxes(image, doc) -> np.ndarray`: desenha todas as caixas com uma etiqueta
  `[id]`, para o mapa de deteções do segundo separador.
- `highlight(image, doc, line_ids, alpha=0.25) -> np.ndarray`: linhas não citadas a cinza
  esbatido e linhas citadas com preenchimento semitransparente, bordo sólido e etiqueta. É o
  que transforma um `line_id` numa prova visível.
- `crop(image, doc, line_ids, pad_px=16) -> np.ndarray | None`: recorte da união das caixas com
  margem, para a caixa de detalhe do primeiro separador. Devolve `None` quando nenhum `line_id`
  existe, e a interface esconde o painel.
- Nenhuma das funções altera a imagem de entrada: todas devolvem uma cópia BGR, convertendo
  imagens em escala de cinzentos quando necessário.

**WP-12 `src/extraction/qa_assistant.py`** (implementado)

- `ask(question: str, doc: OcrDocument, client: LLMClient | None = None, provider: str = "llm_local") -> Answer`:
  submete a pergunta em linguagem natural e o documento OCR ao LLM, exigindo JSON estrito com
  `answer`, `status` (`ENCONTRADO` vs. `AUSENTE NO DOCUMENTO`) e `linhas`.
- Quando o modelo indica `ENCONTRADO` mas não cita linhas válidas, `_locate_supporting_lines` re-ancora
  deterministicamente as linhas de suporte extraindo tokens limpos de pontuação e filtrados por
  stopwords da resposta e da pergunta.
- Se o documento não contiver linhas de texto, responde imediatamente `AUSENTE NO DOCUMENTO` sem chamar o modelo.

**WP-16 `src/app.py`** (implementado)

- `gr.Blocks` com os três separadores já especificados no `PROJECT_PLAN.md`.
- Separador 1: Leitor e Assistente de Documentos, com controlos OpenCV, seletor de OCR, seletor de extrator,
  e seletor desacoplado do modelo para Q&A (`qa_provider_choice`).
- Separador 2: Pipeline Passo a Passo (imagens bruta, alinhada/recortada, filtrada e caixas delimitadoras).
- Separador 3: Validação e Benchmark, com carregamento do relatório `metrics.csv` com `utf-8-sig` e exibição
  estruturada dos cabeçalhos.
- Caches de OCR e de motores em memória com proteção de concorrência via `threading.Lock()`.
- `demo.queue().launch(share=False)`.

---

## 8. Especificação do Golden Set

### 8.1 Composição

- **20 documentos**, dos quais:
  - 12 direitos.
  - 8 amarrotados e desdobrados.
  - Pelo menos 6 **pares** do mesmo talão em ambos os estados. É isto que isola o efeito do
    vinco em vez de o confundir com a diferença entre comerciantes.
  - Pelo menos 3 sem NIF de cliente.
  - Pelo menos 2 de restauração (talão curto, sem IVA discriminado) e 2 de supermercado (talão
    longo, com várias taxas).
  - Pelo menos 2 talões de terminal de pagamento sem QR, para provar que o pipeline não depende
    do atalho do QR.

### 8.2 Convenção de nomes

```text
NN_estado_comerciante.jpg
NN_estado_comerciante_par.jpg
```

Exemplos: `01_direito_pingo_doce.jpg`, `02_amarrotado_pingo_doce_par.jpg`.

**Estado a 2026-11-04:** `data/receipts/` tem 27 imagens recolhidas (`01_direito_controlauto.jpg`,
`02_amachucado_tiendaanimal.jpg`, etc., onde os talões amarrotados usam o prefixo `NN_amachucado_...`
no nome do ficheiro e o grupo canónico `"amarrotado"` na coluna CSV) e `data/ground_truth.csv`
contém os 27 registos com os nomes comerciais de `entidade` alinhados aos cabeçalhos impressos (WP-17
concluído), validados sem erros por `./run check-dataset` e avaliados pelo benchmark (WP-15).

### 8.3 Protocolo de captura

- Fotografar de cima, com o telemóvel paralelo ao papel.
- Luz difusa; evitar flash direto (cria reflexo especular no papel térmico).
- Encher pelo menos 70 % do enquadramento com o talão.
- Incluir sempre a zona do QR e a linha do total.
- Para os pares: fotografar primeiro o talão direito, depois amarrotar, desdobrar e fotografar de
  novo, mantendo um enquadramento semelhante.

### 8.4 `data/ground_truth.csv`

Preenchido à mão a partir da leitura visual de cada imagem. Regras:

- Valores exatamente como impressos, exceto `data` (`AAAA-MM-DD`) e `total` (decimal com ponto,
  sem símbolo).
- Um campo ausente no documento fica **vazio**: não `0`, não `N/A`, não `-`.
- Se um valor estiver ilegível por dano físico, deixar vazio e anotar na coluna de observações
  do ficheiro de trabalho (não no CSV).

### 8.5 Controlo de qualidade

Deve-se revê 100 % das linhas contra as imagens antes do primeiro benchmark. As divergências
resolvem-se sempre a favor da imagem, nunca da memória. Este passo é obrigatório porque uma
verdade de referência errada invalida todas as métricas do relatório.

### 8.6 Mini-apresentação (5 a 7 slides)

1. Porque 20 documentos e não um só.
2. Critérios de estratificação (direito/amarrotado, comerciante, campos ausentes).
3. Protocolo de captura e porque é que a luz difusa importa.
4. Como foi construída a verdade de referência e como foi verificada.
5. Estatísticas do corpus (geradas por `python -m src.evaluation.dataset_stats`).
6. Limitações do corpus (amostra pequena, geograficamente concentrada).

---

## 9. Protocolo de Trabalho e Revisão

### 9.1 Git

- Trunk Based Development: branches curtos a partir de `master`, dois dias no máximo.
- Nomes: `feat/wp-02-normalizers`, `feat/wp-06-tesseract`, `fix/ocr-rotation`.
- Commits pequenos e descritivos, em inglês, no imperativo.
  
### 9.2 Tamanho e forma do merge request

- No máximo ~200 linhas de diff útil.
- Um pacote de trabalho por MR.
- Descrição obrigatória: o que faz, o que não faz, como foi testado, dúvidas em aberto.

### 9.3 Checklist de revisão (usar em todos os MRs)

- [ ] Todos os símbolos públicos têm type hints e docstring no estilo Google.
- [ ] Nenhum `print` em código de biblioteca (só em módulos de linha de comando).
- [ ] Nenhuma leitura de ambiente fora de `src/core/settings.py`.
- [ ] Nenhum caminho absoluto hardcoded.
- [ ] **Os testes cobrem a tabela de aceitação do pacote e passam com `uv run pytest`.**
- [ ] **Os testes novos não acedem à rede, a modelos nem a chaves de API.**
- [ ] `ruff format`, `ruff check`, `pyright` e `pytest` limpos.
- [ ] O diff não toca em ficheiros fora do âmbito declarado.
- [ ] Números mágicos substituídos por constantes nomeadas.

### 9.4 Como dar feedback de código

- Um comentário por problema, com o porquê, não só o quê.
- Sugerir a correção em pseudocódigo quando o problema é conceptual.
- Se o mesmo erro aparecer duas vezes, parar a revisão e explicar em chamada; revisões longas
  por escrito não ensinam.

### 9.5 Se um teste falhar em revisão

Um teste que falha não é motivo para reescrever o teste. Primeiro percebe-se se o teste está
errado ou se expôs um erro no código. Se o teste estava errado, corrige-se e explica-se no MR.
Se expôs um erro no código, o erro é corrigido e o teste fica como regressão. Nunca se apaga
uma asserção para a suite ficar verde.

---

## 10. Ordem de Execução

| Dia | Vitor | José | Raquel |
| :--- | :--- | :--- | :--- |
| 0 | WP-00 estrutura, WP-01 contratos | Ler o direito, preparar o ambiente | Arrancar a recolha de imagens |
| 1 | WP-04 pré-processamento | WP-02 normalizadores (sessão em par, 1 h) | Recolha + `ground_truth.csv` |
| 2 | WP-05 EasyOCR | WP-02 testes e correções | Recolha + `ground_truth.csv` |
| 3 | WP-07 QR, WP-09 regex | WP-03 carregador do golden set | Validação do CSV |
| 4 | WP-08 cliente LLM | WP-06 Tesseract | Mini-apresentação |
| 5 | WP-10 extrator LLM | WP-13 pontuação | Revisão da apresentação |
| 6 | WP-11 evidência, WP-12 Q&A | WP-13 testes, WP-14 estatística | Disponível para apoio |
| 7 | WP-16 Gradio (separadores 1 a 3) | WP-15 runner do benchmark | Disponível para apoio |
| 8 | Revisão de código e integração | WP-15 testes e execução | Disponível para apoio |
| 9 | Integração e benchmark completo | Apoio ao benchmark | Apoio ao benchmark |
| 10 | Congelamento do código | Revisão final dos testes | Fecho dos slides |

Notas:

- O Dia 1 inclui uma sessão em par de uma hora com o José para estabelecer o padrão de código e
  ensinar a correr a suite. Sem isso, os pacotes seguintes geram retrabalho.
- A partir do Dia 7 o José entra em modo de apoio: corre o benchmark, verifica resultados, ajuda
  no relatório.
- Se o José atrasar, o Vitor absorve o WP-06 e o WP-13 sem parar o caminho crítico.

---

## 11. Riscos e Mitigação

| Risco | Impacto | Mitigação |
| :--- | :--- | :--- |
| `easyocr` incompatível com a versão de Python escolhida | Bloqueia o pipeline | Fixar 3.12; validar com `uv add easyocr` no Dia 0, antes de escrever código |
| Pesos do EasyOCR descarregados da internet no primeiro uso | A demonstração offline falha | Pré-aquecer e documentar o diretório `~/.EasyOCR`; correr uma vez com rede antes da aula |
| `llama-server` devolve JSON válido com as chaves erradas | A extração falha em silêncio | Descrever o schema no prompt; validar o JSON e rejeitar chaves desconhecidas |
| Modelo de 3B não cabe nos 2 GB de VRAM | Latência alta na demonstração | Offload parcial (`-ngl`), contexto 8192 e cache de OCR para não repetir a fase cara |
| Golden set com erros | Todas as métricas do relatório ficam inválidas | Revisão a 100 %  antes do primeiro benchmark |
| Imagens contêm NIF e fragmentos de cartão | Exposição de RGPD no GitLab | Manter o repositório privado; opcionalmente mascarar os dígitos do cartão nas imagens; não publicar o corpus fora do grupo |
| Tesseract sem o pacote de língua portuguesa | OCR comparativo degradado | Instalar `tesseract-ocr-por` e documentar no README |
| Gradio muda de API entre versões | A interface quebra | Fixar a versão no `pyproject.toml` e não atualizar depois do Dia 8 |
| Suite de testes lenta ou dependente de modelos | O professor não a consegue correr | Nenhum teste pode carregar EasyOCR, modelos ou rede; alvo de execução abaixo de 10 segundos |
| José bloqueado sem saber a quem perguntar | Atraso silencioso | Regra explícita: bloqueado mais de 30 minutos, mensagem ao Vitor; resposta no mesmo dia |

---

## 12. Critérios de Aceitação do Projeto

O projeto está concluído quando, cumulativamente:

1. A aplicação arranca com `uv run python -m src.app` e processa uma imagem nova de ponta a
   ponta, sem rede.
2. Os cinco campos são extraídos pelos três métodos (regex, LLM local, LLM na cloud) e a tabela
   mostra o estado de validação e o valor do QR quando existir.
3. O assistente responde a perguntas, distingue `ENCONTRADO` de `AUSENTE NO DOCUMENTO` e
   destaca a região da imagem que sustenta a resposta.
4. O benchmark corre sobre os 20 documentos e produz `results/metrics.csv` e
   `results/metrics.md` com a desagregação direito/amarrotado e por método.
5. `uv run pytest`, `ruff` e `pyright` passam sem erros, e a suite completa corre em menos de
   10 segundos sem rede.
6. O relatório consegue citar, para cada afirmação quantitativa, um ficheiro em `results/`.
