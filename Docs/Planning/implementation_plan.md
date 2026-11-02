# Plano de Implementação — Leitor Inteligente de Documentos Visuais

Documento de engenharia que transforma o `PROJECT_PLAN.md` em trabalho executável. Define a
arquitetura de código, os contratos de dados, a divisão de trabalho, o protocolo de revisão e
a ordem de execução.

Complementa (não substitui) o `PROJECT_PLAN.md`, que continua a ser o documento de
enquadramento e de argumentação para o relatório.

Data de referência: 2026-10-30. As datas abaixo são relativas (Dia 0 = arranque). Ancorar ao
prazo real de submissão antes de partilhar.

Equipa: Vitor (pipeline), José (camada de dados e métricas), Raquel (corpus e narrativa).

**Estado em 2026-11-02.** WP-00, WP-01, WP-02, WP-04, WP-05, WP-06, WP-07, WP-08, WP-09 e
WP-11 estão fechados e integrados em `master`. O `src/settings.py` foi acrescentado fora da
numeração para centralizar o acesso ao ambiente e aos caminhos. O detalhe do que foi entregue,
e do que diverge do plano original, está na secção 5.1. As restantes secções descrevem o alvo e
continuam válidas para os pacotes por executar.

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
  possam fazer `from src.normalizers import parse_moeda` sem instalar o pacote.
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

**Arranque verde:** o WP-00 inclui `tests/test_smoke.py`, que importa `src.types`, constrói um
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
│   ├── Planning/
│   │   ├── PROJECT_PLAN.md
│   │   └── implementation_plan.md
│   └── Presentation/              # mini-apresentação da Raquel (pendente)
├── data/
│   ├── receipts/                  # 9 imagens presentes; convenção de nomes pendente
│   │   ├── 01_plano_pingo_doce.jpg
│   │   ├── 02_amarrotado_pingo_doce_par.jpg
│   │   └── ...
│   └── ground_truth.csv           # verdade de referência; só cabeçalho por agora
├── models/                        # .gguf (git-ignored)
├── results/                       # saídas do benchmark (versionadas, pequenas)
├── src/
│   ├── __init__.py                # feito
│   ├── types.py                   # contratos de dados          [Vitor]  feito
│   ├── settings.py                # ambiente + caminhos         [Vitor]  feito
│   ├── normalizers.py             # funções puras               [José]   feito
│   ├── golden_set.py              # carregamento + validação    [José]   pendente
│   ├── preprocessing.py           # OpenCV                      [Vitor]  feito
│   ├── ocr_engine.py              # Protocol + EasyOCR          [Vitor]  feito
│   ├── ocr_tesseract.py           # backend Tesseract           [José]   feito
│   ├── qr_reader.py               # QR da AT                    [Vitor]  feito
│   ├── extractor_regex.py         # baseline                    [Vitor]  feito
│   ├── llm_client.py              # local + cloud               [Vitor]  feito
│   ├── extractor_llm.py           # schema + prompt + parse     [Vitor]  pendente
│   ├── qa_assistant.py            # Q&A com evidência           [Vitor]  pendente
│   ├── evidence.py                # bbox -> recorte/destaque    [Vitor]  feito
│   ├── scoring.py                 # comparação e agregação      [José]   pendente
│   ├── evaluate.py                # runner do benchmark         [Vitor]  pendente
│   ├── dataset_stats.py           # estatística do corpus       [José]   pendente
│   └── app.py                     # Gradio                      [Vitor]  pendente
├── tests/
│   ├── conftest.py                # fixtures partilhadas        [Vitor]  feito
│   ├── test_smoke.py              # arranque verde              [Vitor]  feito
│   ├── test_settings.py           # settings e .env             [Vitor]  feito
│   ├── test_preprocessing.py      # OpenCV e sonda OSD          [Vitor]  feito
│   ├── test_ocr_contract.py       # contrato do OcrEngine       [Vitor]  feito
│   ├── test_qr_reader.py          # QR sintético e parser       [Vitor]  feito
│   ├── test_evidence.py           # destaque e recorte          [Vitor]  feito
│   ├── test_normalizers.py        [José]   feito
│   ├── test_golden_set.py         [José]   pendente
│   ├── test_scoring.py            [José]   pendente
│   ├── test_ocr_tesseract.py      [José]   feito
│   ├── test_llm_client.py         [Vitor]  feito
│   └── test_extractor_regex.py    [Vitor]  feito
├── scripts/                       # atalhos de arranque
│   ├── check_golden_set.sh        # atalho de validação         [Vitor]  pendente
│   └── run_llama_server.sh        # arranque do servidor        [Vitor]  feito
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

`feito` significa integrado em `master` com testes a passar; `pendente`, não iniciado. A árvore
está atualizada a 2026-11-02. O `run_llama_server` existe também em `.ps1` e `.bat`, porque a
equipa trabalha em três sistemas.

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

## 3. Contratos de Dados (`src/types.py`)

Este ficheiro é a peça central da coordenação. É escrito pelo Vitor no Dia 0 e **não muda sem
aviso à equipa**. Todos programam contra ele. A fonte de verdade é o próprio ficheiro, que
documenta cada campo; esta secção resume o que ele contém.

Os tipos são todos `@dataclass(frozen=True, slots=True)` (`OcrEngine` é um `Protocol`), os
identificadores de linha começam em 1 e `status` é
`Literal["ENCONTRADO", "AUSENTE", "INVALIDO"]`. A implementação acrescentou três membros
derivados, sem alterar nenhum campo do rascunho original:

| Tipo | Acrescento |
| :--- | :--- |
| `BBox` | propriedades `width` e `height` |
| `OcrDocument` | `as_prompt_text()`, que numera as linhas como `[01] ...` para o prompt do LLM |
| `Extraction` | propriedade `fields`, que devolve os cinco campos por nome de coluna |

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
   └─ evidence.draw_all_ocr_boxes(image, OcrDocument)               -> np.ndarray
      evidence.highlight(image, OcrDocument, line_ids, alpha=0.25)  -> np.ndarray
      evidence.crop(image, OcrDocument, line_ids, pad_px=16)        -> np.ndarray | None
```

`PreprocessResult` e as respetivas opções vivem em `preprocessing.py`, não em `types.py`,
porque só esse módulo as usa. `crop` devolve `None` quando nenhum dos `line_ids` existe no
documento, para que a interface possa simplesmente esconder o painel de recorte.

---

## 5. Pacotes de Trabalho

| ID | Título | Dono | Depende de | Tamanho | Ficheiros | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| WP-00 | Estrutura inicial do repositório | Vitor | — | 3 h | `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`, `tests/conftest.py`, `tests/test_smoke.py` | feito |
| WP-01 | Contratos de dados | Vitor | WP-00 | 2 h | `src/types.py` | feito |
| WP-02 | Normalizadores puros + testes | **José** | WP-01 | 4 a 6 h | `src/normalizers.py`, `tests/test_normalizers.py` | feito |
| WP-03 | Carregador e validador do golden set | **José** | WP-01, WP-02 | 4 a 6 h | `src/golden_set.py`, `tests/test_golden_set.py` | pendente |
| WP-04 | Pré-processamento OpenCV | Vitor | WP-01 | 8 h | `src/preprocessing.py` | feito |
| WP-05 | Motor OCR EasyOCR | Vitor | WP-01, WP-04 | 6 h | `src/ocr_engine.py` | feito |
| WP-06 | Backend Tesseract | **José** | WP-05 | 3 a 4 h | `src/ocr_tesseract.py` | feito |
| WP-07 | Leitor de QR da AT | Vitor | WP-04 | 3 h | `src/qr_reader.py` | feito |
| WP-08 | Cliente LLM unificado | Vitor | WP-01 | 6 h | `src/llm_client.py` | feito |
| WP-09 | Extrator por regex (baseline) | Vitor | WP-02, WP-05 | 5 h | `src/extractor_regex.py`, `tests/test_extractor_regex.py` | feito |
| WP-10 | Extrator por LLM | Vitor | WP-08, WP-09 | 6 h | `src/extractor_llm.py` | pendente |
| WP-11 | Evidência visual | Vitor | WP-01 | 3 h | `src/evidence.py` | feito |
| WP-12 | Assistente de Q&A com evidência | Vitor | WP-08, WP-11 | 5 h | `src/qa_assistant.py` | pendente |
| WP-13 | Pontuação e agregação | **José** | WP-02, WP-03 | 5 a 7 h | `src/scoring.py`, `tests/test_scoring.py` | pendente |
| WP-14 | Estatística do corpus | **José** | WP-03 | 2 a 3 h | `src/dataset_stats.py` | pendente |
| WP-15 | Runner do benchmark | Vitor | WP-13 | 5 h | `src/evaluate.py` | pendente |
| WP-16 | Aplicação Gradio | Vitor | WP-04 a WP-12 | 10 h | `src/app.py` | pendente |
| WP-17 | Golden set: imagens + CSV | **Raquel** | — | 3 a 4 h | `data/receipts/*`, `data/ground_truth.csv` | em curso |
| WP-18 | Mini-apresentação do golden set | **Raquel** | WP-17 | 3 h | `Docs/Presentation/` | pendente |

O caminho crítico é WP-00 → WP-01 → WP-04 → WP-05 → WP-09 → WP-10 → WP-16. WP-09 não
figurava na versão original desta linha, mas o WP-10 depende dele desde a primeira tabela: o
extrator por LLM reutiliza o contrato e os casos de teste do baseline.

### 5.1 Estado das entregas e desvios face ao plano

Fechados e integrados em `master` a 2026-11-02: WP-00, WP-01, WP-02, WP-04, WP-05, WP-06,
WP-07, WP-08, WP-09 e WP-11. A suite tem 172 testes offline que correm em cerca de 2 s, mais um
teste `slow` com o EasyOCR real, deselecionado por omissão.

Desvios que valem registo, porque afetam os pacotes seguintes:

- **`src/settings.py` (fora da numeração).** Centraliza `os.getenv`, o carregamento do `.env` e
  a resolução de todos os caminhos, para cumprir a regra "nenhuma leitura de ambiente fora de
  `settings.py`". `load_settings(repo_root=...)` constrói um `Settings` isolado para um
  diretório temporário, que é o que torna os testes de CLI possíveis sem escrever no
  repositório; `get_settings()` é `lru_cache`ado e não deve ser usado em testes.
- **Convenção de CLI.** Todos os módulos correm como `python -m src.<modulo>` e expõem
  `main(argv=None, settings_override=None)`. Os imports pesados (Pillow, EasyOCR, `settings`)
  ficam adiados para `main` ou para o construtor do motor. Isso mantém o import barato e é o
  que permite correr a suite em 0,3 s.
- **Validação de imagens.** Todas as funções públicas que recebem uma imagem levantam
  `ValueError("Input image must not be empty or zero-sized.")` e devolvem sempre uma cópia, sem
  mutar a entrada. Os testes de `evidence.py` verificam a não-destrutividade.
- **WP-04:** além da heurística de proporção, entrou uma sonda de orientação por Tesseract OSD
  (`--psm 0`, timeout de 5 s, devolve `0` em qualquer falha) e o recorte passou a fazer warp de
  perspetiva sobre o maior quadrilátero. Ambas são desligáveis em `PreprocessOptions`.
- **WP-05:** `EasyOCREngine` aceita `reader=` para injetar um leitor falso nos testes e ordena
  as linhas por `(y1, x1)` antes de as numerar.
- **WP-07:** a descodificação tem cinco estágios, incluindo a procura de blobs quadrados com
  reparação das riscas verticais deixadas por cabeças térmicas avariadas. O parser mantém as
  chaves cruas da AT e acrescenta as alias usadas pelo resto do pipeline.
- **WP-11:** a API ficou `draw_all_ocr_boxes`, `highlight` e `crop`, com `crop` a devolver
  `None` quando nenhum `line_id` existe. `highlight` desenha as linhas não citadas em cinza
  esbatido, para que a evidência ativa salte à vista.
- **WP-02:** os normalizadores são puros e não leem ficheiros, ambiente nem desenham nada.
  `parse_data` lê datas com hífen como `YY-MM-DD`, o que resolve o corpus mas cria ambiguidade
  com `DD-MM-YY`; `parse_moeda` devolve o **primeiro** número da string, não o último.
- **WP-06:** `TesseractEngine` agrega caixas de palavra por `(block_num, par_num, line_num)`,
  descarta `conf < 0` e restaura o `pytesseract.pytesseract.tesseract_cmd` global no fim, para
  não deixar estado entre testes.
- **WP-08:** `llm_client` expõe `OpenAICompatibleClient` (que serve o `llama-server` e a OpenAI
  com o mesmo caminho de código), `GeminiClient` e `FailoverLLMClient`. O failover marca o
  provedor efetivo em `last_effective_provider`, e `build_client` aceita clientes injetados
  para os testes nunca tocarem na rede.
- **WP-09:** ver a secção 6, WP-09, para as regras do baseline e as três armadilhas que o
  pacote fecha (sentinela do consumidor final, NIF duplicado e total com quantidade à frente).

Continua em aberto e bloqueia o benchmark: o golden set (WP-17) tem 9 imagens em
`data/receipts/` e um `data/ground_truth.csv` só com cabeçalho. Sem verdade de referência não
há `scoring.py` nem `evaluate.py` que produzam números citáveis. O WP-09, já fechado, foi
validado à custa de documentos de OCR sintéticos: a precisão do baseline sobre o corpus real só
pode ser medida depois de o WP-17 entregar a verdade de referência.

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

### WP-09 — Extrator por regex (`src/extractor_regex.py`)

Este pacote não tinha especificação detalhada na primeira versão do plano, só a linha da
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

**WP-04 `preprocessing.py`** (implementado)

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

**WP-05 `ocr_engine.py`** (implementado)

- `EasyOCREngine(langs=("pt", "en"), gpu=False, warmup=True, reader=None)`; o `reader` é o
  ponto de injeção usado pelos testes, que assim nunca carregam pesos.
- Pré-aquecimento com uma imagem sintética no arranque, para que a primeira inferência real
  não pague o custo de carregar os pesos.
- Mapear `readtext(detail=1)` para `OcrLine` com `id` sequencial: caixas clampadas à imagem,
  texto vazio e caixas de área nula descartados, e ordenação por `(y1, x1)` antes de numerar.

**WP-07 `qr_reader.py`** (implementado)

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

**WP-11 `evidence.py`** (implementado)

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

**WP-16 `app.py`** (pendente)

- `gr.Blocks` com os três separadores já especificados no `PROJECT_PLAN.md`.
- O mapa de deteções usa `evidence.draw_all_ocr_boxes`; a prova visual usa
  `evidence.highlight` e `evidence.crop`.
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

**Estado a 2026-11-01:** `data/receipts/` tem 9 imagens com nomes numéricos
(`5990024681546780813.jpg`, ...), ainda fora desta convenção, e `data/ground_truth.csv` só tem
o cabeçalho. Renomear as imagens e preencher a verdade de referência são passos do WP-17; até
lá, o benchmark e a pontuação não podem produzir resultados. O validador do WP-03 só exige que
o `filename` do CSV exista na pasta, por isso a convenção não bloqueia código, apenas a
legibilidade do corpus para o relatório.

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
