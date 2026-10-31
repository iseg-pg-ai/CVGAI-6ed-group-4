# Plano de Implementação — Leitor Inteligente de Documentos Visuais

Documento de engenharia que transforma o `PROJECT_PLAN.md` em trabalho executável. Define a
arquitetura de código, os contratos de dados, a divisão de trabalho, o protocolo de revisão e
a ordem de execução.

Complementa (não substitui) o `PROJECT_PLAN.md`, que continua a ser o documento de
enquadramento e de argumentação para o relatório.

Data de referência: 2026-10-30. As datas abaixo são relativas (Dia 0 = arranque). Ancorar ao
prazo real de submissão antes de partilhar.

Equipa: Vitor (pipeline), José (camada de dados e métricas), Raquel (corpus e narrativa).

---

## 0. Princípios de Engenharia

1. **Contratos primeiro, implementação depois.** O ficheiro `src/types.py` é escrito e revisto
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
| LLM/VLM | `llama.cpp` (`llama-server`) local + Gemini/OpenAI na cloud | Idem |
| Interface | Gradio | Idem |
| QR | `opencv-python-headless` (`cv2.QRCodeDetector`) com recurso alternativo a `pyzbar` | Sem dependência de serviços externos. Ver nota em 1.1: `contrib` não é necessário e conflitua com o EasyOCR. |

### 1.1 Dependências de sistema (fora do `uv`)

- `tesseract-ocr` e `tesseract-ocr-por` (pacotes do sistema, não do pip).
- Binário `llama-server` compilado com backend Vulkan (funciona na GPU antiga do portátil e na
  RX 9070 XT) ou descarregado dos releases oficiais.

Ambos têm de ficar documentados no `README.md` com os comandos de instalação por distribuição,
porque o José precisa deles para correr a suite localmente.

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

**Configuração em `pyproject.toml` (criada no WP-00):**

```toml
[project]
name = "cvgai-6ed-group-4"
version = "0.1.0"
requires-python = ">=3.12"

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
addopts = "-q --strict-markers"
```

Notas sobre estas opções:

- `testpaths = ["tests"]` evita que o `pytest` ande a varrer `class-material/`, `data/` e
  `.venv/`. Sem isto, a recolha de testes tenta importar ficheiros que não são testes e a
  suite falha por motivos errados.
- `pythonpath = ["."]` põe a raiz do repositório no caminho de importação, para que os testes
  possam fazer `from src.normalizers import parse_moeda` sem instalar o pacote.
- `--strict-markers` faz falhar um teste que use um marcador não declarado. É a forma de
  impedir que um `@pytest.mark.skip` mal escrito passe silenciosamente.

**Comandos do dia a dia:**

```bash
uv sync                                  # instala dependências, incluindo o grupo dev
uv run pytest                            # corre a suite toda
uv run pytest tests/test_normalizers.py -q          # um ficheiro
uv run pytest -k "nif" -q                           # filtra por nome
uv run pytest --cov=src --cov-report=term-missing   # com cobertura
```

**Regras a que os testes têm de obedecer:**

1. Um teste nunca acede à rede, a uma GPU, a uma chave de API ou a um modelo descarregado.
2. Um teste nunca depende da ordem de execução nem do estado deixado por outro teste.
3. Tudo o que é temporário vai para `tmp_path`; nada escreve dentro do repositório.
4. Os testes de OCR não correm o EasyOCR real. Validam o contrato (`OcrDocument`, caixas
   dentro dos limites da imagem) com imagens sintéticas e, quando muito, com o Tesseract.
5. Um teste que falha tem de dizer o que se esperava e o que se obteve. `assert resultado ==
   esperado` sem contexto é uma perda de tempo para quem revê.

**Fixtures partilhadas em `tests/conftest.py`:**

```python
@pytest.fixture
def receipts_tmp(tmp_path: Path) -> Path:
    """Cria um diretório de imagens com ficheiros vazios para validação do golden set."""

@pytest.fixture
def ground_truth_valido(tmp_path: Path, receipts_tmp: Path) -> Path:
    """Escreve um CSV mínimo mas válido, coerente com receipts_tmp."""

@pytest.fixture
def extracao_vazia() -> Extraction:
    """Extraction com os cinco campos em AUSENTE, para testes de pontuação."""
```

**Arranque verde:** o WP-00 inclui `tests/test_smoke.py` com um único teste que importa
`src.types` e constrói um `BBox`. Serve para provar que o ambiente, a configuração do
`pytest` e o caminho de importação estão corretos antes de existir lógica de negócio. Um
`pytest` que falha no Dia 0 é um problema de ambiente; um que falha no Dia 8 é um problema de
código, e nessa altura já não há tempo para distinguir os dois.

**Para quem avalia:** numa máquina já preparada, a suite completa corre com um único comando
(`uv run pytest`), sem rede e sem modelos. O resultado é determinístico.

---

## 2. Estrutura do Repositório

```text
CVGAI-6ed-group-4/
├── Docs/
│   ├── Planning/
│   │   ├── PROJECT_PLAN.md
│   │   └── implementation_plan.md
│   └── Presentation/              # mini-apresentação da Raquel
├── data/
│   ├── receipts/                  # imagens numeradas do golden set
│   │   ├── 01_plano_pingo_doce.jpg
│   │   ├── 02_amarrotado_pingo_doce_par.jpg
│   │   └── ...
│   └── ground_truth.csv           # verdade de referência (Raquel)
├── models/                        # .gguf (git-ignored)
├── results/                       # saídas do benchmark (versionadas, pequenas)
├── src/
│   ├── __init__.py
│   ├── types.py                   # contratos de dados          [Vitor]
│   ├── settings.py                # ambiente + caminhos         [Vitor]
│   ├── normalizers.py             # funções puras               [José]
│   ├── golden_set.py              # carregamento + validação    [José]
│   ├── preprocessing.py           # OpenCV                      [Vitor]
│   ├── ocr_engine.py              # Protocol + EasyOCR          [Vitor]
│   ├── ocr_tesseract.py           # backend Tesseract           [José]
│   ├── qr_reader.py               # QR da AT                    [Vitor]
│   ├── extractor_regex.py         # baseline                    [Vitor]
│   ├── llm_client.py              # local + cloud               [Vitor]
│   ├── extractor_llm.py           # schema + prompt + parse     [Vitor]
│   ├── qa_assistant.py            # Q&A com evidência           [Vitor]
│   ├── evidence.py                # bbox -> recorte/destaque    [Vitor]
│   ├── scoring.py                 # comparação e agregação      [José]
│   ├── evaluate.py                # runner do benchmark         [Vitor]
│   ├── dataset_stats.py           # estatística do corpus       [José]
│   └── app.py                     # Gradio                      [Vitor]
├── tests/
│   ├── conftest.py                # fixtures partilhadas        [Vitor]
│   ├── test_smoke.py              # arranque verde              [Vitor]
│   ├── test_normalizers.py        [José]
│   ├── test_golden_set.py         [José]
│   ├── test_scoring.py            [José]
│   └── test_ocr_contract.py       [Vitor]
├── scripts/
│   ├── check_golden_set.sh        # atalho de validação         [Vitor]
│   └── run_llama_server.sh        # arranque do servidor        [Vitor]
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

### `.gitignore` mínimo

```text
.venv/
__pycache__/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/
models/
data/receipts/*.jpg   # opcional: ver o risco de RGPD na secção 11
results/*.json
.env
```

---

## 3. Contratos de Dados (`src/types.py`)

Este ficheiro é a peça central da coordenação. É escrito pelo Vitor no Dia 0 e **não muda sem
aviso à equipa**. Todos programam contra ele.

```python
"""Shared data contracts for the whole pipeline."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal, Protocol

import numpy as np

Status = Literal["ENCONTRADO", "AUSENTE", "INVALIDO"]


@dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box in pixels, in the processed image reference frame."""
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class OcrLine:
    """Recognised text line with a stable identifier and its geometry."""
    id: int                 # 1-based; the LLM cites this to ground an answer
    text: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True, slots=True)
class OcrDocument:
    """Complete result of one OCR engine over one image."""
    lines: list[OcrLine]
    engine: str             # "easyocr" | "tesseract"
    elapsed_ms: float
    image_size: tuple[int, int]   # (width, height)


class OcrEngine(Protocol):
    """Interface every OCR engine must satisfy."""
    name: str

    def read(self, image: np.ndarray) -> OcrDocument: ...


@dataclass(frozen=True, slots=True)
class FieldValue:
    """Extracted value for one field, with status and supporting evidence."""
    value: str | None
    status: Status
    line_id: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Extraction:
    """The five fields required by the assignment, plus run metadata."""
    entidade: FieldValue
    nif_comerciante: FieldValue
    nif_cliente: FieldValue
    data: FieldValue
    total: FieldValue
    method: str             # "regex" | "llm_local" | "llm_gemini" | "llm_openai"
    elapsed_ms: float
    raw_response: str | None = None


@dataclass(frozen=True, slots=True)
class GroundTruthRecord:
    """One already validated and typed row of ground_truth.csv."""
    filename: str
    grupo: Literal["plano", "amarrotado"]
    entidade: str | None
    nif_comerciante: str | None
    nif_cliente: str | None
    data: date | None
    total: Decimal | None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Comparison outcome for one field between prediction and truth."""
    filename: str
    grupo: str
    campo: str
    esperado: str | None
    obtido: str | None
    acertou: bool
    falso_positivo: bool     # invented a value where none existed
    omissao: bool            # returned nothing where a value existed


@dataclass(frozen=True, slots=True)
class Answer:
    """Assistant answer, with presence status and citable evidence."""
    question: str
    answer: str
    status: Literal["ENCONTRADO", "AUSENTE NO DOCUMENTO"]
    line_ids: list[int]
    provider: str
```

**Regra de ouro:** se um pacote de trabalho precisar de um campo novo nestes tipos, isso é uma
alteração de contrato e é feita pelo Vitor, não pelo autor do pacote.

---

## 4. Interface do Pipeline (o que cada fase consome e produz)

```text
imagem (np.ndarray BGR)
   │
   ├─ preprocessing.preprocess(img, opts) -> PreprocessResult
   │     .image         imagem alinhada, recortada e com CLAHE
   │     .steps         dict[str, np.ndarray]  (bruta, rodada, recortada, filtrada)
   │     .rotation_deg  rotação aplicada
   │
   ├─ ocr_engine.read(preprocessed.image) -> OcrDocument
   │
   ├─ qr_reader.read(preprocessed.image) -> dict[str, str] | None
   │
   ├─ extractor_*.extract(OcrDocument, qr) -> Extraction
   │
   ├─ qa_assistant.ask(question, OcrDocument) -> Answer
   │
   └─ evidence.highlight(image, OcrDocument, line_ids) -> np.ndarray
      evidence.crop(image, OcrDocument, line_ids)      -> np.ndarray
```

`PreprocessResult` e as respetivas opções vivem em `preprocessing.py`, não em `types.py`,
porque só esse módulo as usa.

---

## 5. Pacotes de Trabalho

| ID | Título | Dono | Depende de | Tamanho | Ficheiros |
| :--- | :--- | :--- | :--- | :--- | :--- |
| WP-00 | Estrutura inicial do repositório | Vitor | — | 3 h | `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`, `tests/conftest.py`, `tests/test_smoke.py` |
| WP-01 | Contratos de dados | Vitor | WP-00 | 2 h | `src/types.py` |
| WP-02 | Normalizadores puros + testes | **José** | WP-01 | 4 a 6 h | `src/normalizers.py`, `tests/test_normalizers.py` |
| WP-03 | Carregador e validador do golden set | **José** | WP-01, WP-02 | 4 a 6 h | `src/golden_set.py`, `tests/test_golden_set.py` |
| WP-04 | Pré-processamento OpenCV | Vitor | WP-01 | 8 h | `src/preprocessing.py` |
| WP-05 | Motor OCR EasyOCR | Vitor | WP-01, WP-04 | 6 h | `src/ocr_engine.py` |
| WP-06 | Backend Tesseract | **José** | WP-05 | 3 a 4 h | `src/ocr_tesseract.py` |
| WP-07 | Leitor de QR da AT | Vitor | WP-04 | 3 h | `src/qr_reader.py` |
| WP-08 | Cliente LLM unificado | Vitor | WP-01 | 6 h | `src/llm_client.py` |
| WP-09 | Extrator por regex (baseline) | Vitor | WP-02, WP-05 | 5 h | `src/extractor_regex.py` |
| WP-10 | Extrator por LLM | Vitor | WP-08, WP-09 | 6 h | `src/extractor_llm.py` |
| WP-11 | Evidência visual | Vitor | WP-01 | 3 h | `src/evidence.py` |
| WP-12 | Assistente de Q&A com evidência | Vitor | WP-08, WP-11 | 5 h | `src/qa_assistant.py` |
| WP-13 | Pontuação e agregação | **José** | WP-02, WP-03 | 5 a 7 h | `src/scoring.py`, `tests/test_scoring.py` |
| WP-14 | Estatística do corpus | **José** | WP-03 | 2 a 3 h | `src/dataset_stats.py` |
| WP-15 | Runner do benchmark | Vitor | WP-13 | 5 h | `src/evaluate.py` |
| WP-16 | Aplicação Gradio | Vitor | WP-04 a WP-12 | 10 h | `src/app.py` |
| WP-17 | Golden set: imagens + CSV | **Raquel** | — | 3 a 4 h | `data/receipts/*`, `data/ground_truth.csv` |
| WP-18 | Mini-apresentação do golden set | **Raquel** | WP-17 | 3 h | `Docs/Presentation/` |

O caminho crítico é WP-00 → WP-01 → WP-04 → WP-05 → WP-10 → WP-16.

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

### WP-02 — Normalizadores puros (`src/normalizers.py`)

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

def parse_nif(raw: str) -> str | None:
    """Extract nine digits from 'NIF: 518396894', 'NIF 518 396 894', '518396894'."""

def validar_nif(nif: str) -> bool:
    """Validate the modulo-11 check digit and the permitted first digit."""

def normalizar_entidade(raw: str) -> str:
    """Canonical form for comparison: lowercase, no accents, no punctuation,
    no corporate suffixes (lda, sa, unipessoal), whitespace collapsed."""

def valores_equivalentes(campo: str, esperado: str | None, obtido: str | None) -> bool:
    """Compare two already-converted values using the per-field rule.

    Rules: 'total' compares Decimals; 'data' compares dates; 'nif_*' compares
    nine-digit strings; 'entidade' compares canonical forms.
    Two Nones are equivalent.
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

### WP-03 — Carregador e validador do golden set (`src/golden_set.py`)


**Contrato de entrada (`data/ground_truth.csv`), colunas obrigatórias:**

```text
filename,grupo,entidade,nif_comerciante,nif_cliente,data,total
```

- `filename` — nome do ficheiro dentro de `data/receipts/`, com extensão.
- `grupo` — `plano` ou `amarrotado`.
- `entidade` — designação como aparece no talão; vazio se ilegível.
- `nif_comerciante` — nove dígitos; vazio apenas se ausente no documento.
- `nif_cliente` — nove dígitos; vazio quando o talão não tem contribuinte.
- `data` — `AAAA-MM-DD`.
- `total` — decimal com ponto (`16.10`), sem símbolo de moeda.

**Assinaturas:**

```python
def carregar_ground_truth(path: Path) -> list[GroundTruthRecord]:
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
3. `grupo` em `{"plano", "amarrotado"}`.
4. `nif_comerciante` e `nif_cliente`, quando preenchidos, têm nove dígitos e passam
   `validar_nif`.
5. `data`, quando preenchida, é interpretável e não está no futuro.
6. `total`, quando preenchido, é interpretável e maior que zero.
7. Nenhuma linha totalmente vazia; nenhuma coluna extra.

**Critério de aceitação:** um teste que constrói CSVs inválidos em `tmp_path` (ficheiro
inexistente, NIF inválido, data no futuro, duplicado, coluna em falta) e verifica que
`validar_ground_truth` devolve o erro correspondente. Um teste contra o CSV real do repositório
que espera uma lista vazia.

---

### WP-06 — Backend Tesseract (`src/ocr_tesseract.py`)

**Assinatura:**

```python
class TesseractEngine:
    """Tesseract-based OCR engine, aggregating word boxes into lines."""

    name = "tesseract"

    def __init__(self, lang: str = "por", psm: int = 6) -> None: ...

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

### WP-13 — Pontuação e agregação (`src/scoring.py`)

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

def escrever_tabelas(resultados: list[MatchResult], destino: Path) -> None:
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

### WP-14 — Estatística do corpus (`src/dataset_stats.py`)

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
plano         12        12        12       7      12     12
amarrotado     8         7         8       5       8      8
```

---

## 7. Especificação dos Pacotes da pipeline

**WP-04 `preprocessing.py`**

- `resize_max_side(img, 1280)` antes de tudo.
- Rotação: heurística por `w > h` para talões verticais, mais sobreposições 0/90/180/270.
- Recorte: escala de cinzentos → desfoque gaussiano → Canny → contornos → maior quadrilátero
  com área superior a 20 % da imagem; recurso alternativo à imagem inteira.
- Normalização de iluminação: `gray / close(gray, kernel=25x25) * 255`.
- CLAHE com `clipLimit=2.0`, `tileGridSize=(8, 8)`.
- `steps` devolve todas as imagens intermédias para o segundo separador.

**WP-05 `ocr_engine.py`**

- `EasyOCREngine(langs=("pt", "en"), gpu=False)`.
- Pré-aquecimento com uma imagem sintética no arranque, para que a primeira inferência real
  não pague o custo de carregar os pesos.
- Mapear `readtext(detail=1)` para `OcrLine` com `id` sequencial.

**WP-07 `qr_reader.py`**

- `cv2.QRCodeDetector().detectAndDecode` e, em alternativa, `pyzbar.decode`.
- Interpretar o conteúdo da AT: pares `CHAVE:VALOR` separados por `*`.
- Devolver um dicionário com, no mínimo: `A` (NIF emitente), `B` (NIF adquirente), `F` (data
  `AAAAMMDD`), `H` (ATCUD), `O` (total).
- Uso: validação cruzada e coluna extra na interface. **Nunca** substitui o OCR, porque os
  talões de terminal de pagamento não têm QR.

**WP-08 `llm_client.py`**

```python
class LLMClient(Protocol):
    provider: str
    def complete(self, system: str, user: str, schema: dict | None) -> str: ...

def build_client(provider: str) -> LLMClient: ...
```

- `llama-server` e OpenAI partilham o mesmo caminho de código (`openai.OpenAI(base_url=...)`).
- O Gemini usa `google-genai` e converte o schema para `response_schema`.
- Failover: em erro de rede, erro de quota ou timeout, cai para o cliente local e marca o
  `provider` efetivo na resposta.

**WP-10 `extractor_llm.py`**

- Prompt com o texto indexado linha a linha (`[01] ...`), tal como planeado.
- Schema plano, sem uniões de tipo (o conversor `json-schema-to-grammar` do `llama.cpp` só
  suporta um subconjunto): todos os campos são `string`, com `""` para ausente, mais um objeto
  `presente` com booleanos.
- **Ponto confirmado:** o schema em `response_format` **não é injetado no prompt** pelo
  `llama-server`. A estrutura tem de ser descrita em texto no prompt, caso contrário o modelo
  produz JSON válido com as chaves erradas.
- Pós-processamento: validar o JSON, mapear `line_id` para `FieldValue` e correr `validar_nif`
  para rebaixar NIFs inválidos a `INVALIDO`.

**WP-16 `app.py`**

- `gr.Blocks` com os três separadores já especificados no `PROJECT_PLAN.md`.
- Cache de OCR por `sha256(bytes) + opções de pré-processamento`, para o Q&A não repetir a fase
  mais cara.
- `demo.queue().launch(share=False)`.

---

## 8. Especificação do Golden Set

### 8.1 Composição

- **20 documentos**, dos quais:
  - 12 planos.
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

Exemplos: `01_plano_pingo_doce.jpg`, `02_amarrotado_pingo_doce_par.jpg`.

### 8.3 Protocolo de captura

- Fotografar de cima, com o telemóvel paralelo ao papel.
- Luz difusa; evitar flash direto (cria reflexo especular no papel térmico).
- Encher pelo menos 70 % do enquadramento com o talão.
- Incluir sempre a zona do QR e a linha do total.
- Para os pares: fotografar primeiro o talão plano, depois amarrotar, desdobrar e fotografar de
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
2. Critérios de estratificação (plano/amarrotado, comerciante, campos ausentes).
3. Protocolo de captura e porque é que a luz difusa importa.
4. Como foi construída a verdade de referência e como foi verificada.
5. Estatísticas do corpus (geradas por `python -m src.dataset_stats`).
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
- [ ] Nenhuma leitura de ambiente fora de `settings.py`.
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
| 0 | WP-00 estrutura, WP-01 contratos | Ler o plano, preparar o ambiente | Arrancar a recolha de imagens |
| 1 | WP-04 pré-processamento | WP-02 normalizadores (sessão em par, 1 h) | Recolha + `ground_truth.csv` |
| 2 | WP-05 EasyOCR | WP-02 testes e correções | Recolha + `ground_truth.csv` |
| 3 | WP-07 QR, WP-09 regex | WP-03 carregador do golden set | Validação do CSV |
| 4 | WP-08 cliente LLM | WP-06 Tesseract | Mini-apresentação |
| 5 | WP-10 extrator LLM | WP-13 pontuação | Revisão da apresentação |
| 6 | WP-11 evidência, WP-12 Q&A | WP-13 testes, WP-14 estatística | Disponível para apoio |
| 7 | WP-16 Gradio (separadores 1 e 2) | Buffer / correções | Disponível para apoio |
| 8 | WP-16 separador 3, WP-15 runner | Buffer / correções | Disponível para apoio |
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
   `results/metrics.md` com a desagregação plano/amarrotado e por método.
5. `uv run pytest`, `ruff` e `pyright` passam sem erros, e a suite completa corre em menos de
   10 segundos sem rede.
6. O relatório consegue citar, para cada afirmação quantitativa, um ficheiro em `results/`.
