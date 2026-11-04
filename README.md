# CVGAI 6ª Edição — Grupo 4

Leitor inteligente de talões e faturas simplificadas portuguesas. O sistema recebe a
fotografia de um documento (direito ou amarrotado), aplica pré-processamento com OpenCV,
reconhece o texto por OCR, extrai os cinco campos obrigatórios e responde a perguntas em
linguagem natural com evidência visual citada.

## Âmbito

Campos extraídos de cada documento:

| Campo | Descrição |
| :--- | :--- |
| `entidade` | Designação comercial do emitente |
| `nif_comerciante` | NIF do emitente, validado pelo dígito de controlo |
| `nif_cliente` | NIF do adquirente, quando impresso |
| `data` | Data de emissão |
| `total` | Total pago |

Três métodos de extração correm em paralelo para comparação: expressões regulares
(baseline), LLM local servido por `llama.cpp` e LLM na cloud (Google Gemini).

## Estado atual

O pipeline está construído de ponta a ponta, desde a aquisição e pré-processamento até à
extração estruturada, assistente verificável de Q&A e interface interativa Gradio. O que já
existe e pode ser inspecionado hoje, sem chaves de API e sem modelos descarregados (exceto o
EasyOCR, que descarrega os pesos na primeira execução):

| Módulo | Estado | Pré-visualização |
| :--- | :--- | :--- |
| `src/core/types.py` | Contratos de dados fechados | — |
| `src/core/settings.py` | Configuração e caminhos a partir do `.env` | — |
| `src/core/normalizers.py` | Normalização e comparação de campos | — |
| `src/core/cli_utils.py` | Utilitários partilhados de carregamento e formatação CLI | — |
| `src/evaluation/golden_set.py` | Validação e carregamento do golden set | `uv run python -m src.evaluation.golden_set` |
| `src/vision/preprocessing.py` | Redimensionamento, orientação, recorte, sombras, CLAHE | `uv run python -m src.vision.preprocessing` |
| `src/ocr/__init__.py` | Fábrica unificada de motores OCR (`build_ocr_engine`) | — |
| `src/ocr/easyocr.py` | EasyOCR com caixas delimitadoras e IDs de linha | `uv run python -m src.ocr.easyocr` |
| `src/ocr/tesseract.py` | Motor OCR comparativo, com agregação de palavras em linhas | `uv run python -m src.ocr.tesseract` |
| `src/vision/qr_reader.py` | Leitura do QR da AT para validação cruzada | `uv run python -m src.vision.qr_reader` |
| `src/extraction/llm_client.py` | Clientes local e Gemini com failover | `uv run python -m src.extraction.llm_client` |
| `src/extraction/regex.py` | Extrator por regex (baseline) | `uv run python -m src.extraction.regex` |
| `src/extraction/llm.py` | Extrator por LLM, local ou cloud, com JSON validado | `uv run python -m src.extraction.llm --provider llm_local` |
| `src/vision/evidence.py` | Destaque e recorte da região comprovativa | `uv run python -m src.vision.evidence -l 1,2,3` |
| `src/extraction/qa_assistant.py` | Assistente de Q&A com fundamentação em linhas citadas | `uv run python -m src.extraction.qa_assistant` |
| `src/evaluation/scoring.py` | Pontuação contra a verdade de referência e agregação de métricas | — |
| `src/evaluation/dataset_stats.py` | Estatísticas do corpus por grupo, comerciante e campo | `uv run python -m src.evaluation.dataset_stats` |
| `src/evaluation/evaluate.py` | Runner do benchmark oficial sobre o golden set | `uv run python -m src.evaluation.evaluate` |
| `src/app.py` | Aplicação interativa Gradio (3 separadores) | `uv run python -m src.app` |

Cada módulo CLI opera sobre o primeiro talão de `data/receipts/`; a flag `--file`
escolhe outro ficheiro e `--save` escreve o resultado em disco em vez de abrir a janela.
O comando `./run check-dataset` (ou `run check-dataset` no Windows) corre a validação do golden set e as estatísticas do corpus num só passo.

Por construir: mini-apresentação (WP-18).
O extrator por LLM e o assistente de Q&A podem correr contra um `llama-server` local
(`./run serve` ou `run serve` no Windows) ou contra a chave de API da Google (`GEMINI_API_KEY`) no `.env`.

O corpus conta com 27 imagens recolhidas em `data/receipts/` (`01_...` a `27_...`) e
`data/ground_truth.csv` contém os 27 registos correspondentes validados por `./run check-dataset` (WP-17 concluído, com as designações comerciais alinhadas aos cabeçalhos impressos nos talões).

## Requisitos

- Linux, macOS ou Windows. As três plataformas estão suportadas: o pipeline não usa
  binários específicos de um sistema e todos os caminhos são construídos com `pathlib`.
- `uv` (gestor de pacotes e de versões de Python).
- Python 3.12, obtido automaticamente pelo `uv` a partir de `.python-version`.
- `tesseract-ocr` com o pacote de língua portuguesa, para o motor OCR comparativo e para a
  sonda de orientação.
- `zbar`, para a leitura alternativa de QR. Opcional: sem ela o OpenCV trata do QR.
- `llama-server`, apenas para a extração por LLM local.

### Dependências de sistema

O `uv` resolve o Python e os pacotes, mas o `tesseract`, o `llama-server` e a `zbar` são
componentes do sistema e instalam-se à parte. Nenhum deles é necessário para correr a suite de
testes.

#### Tesseract (motor OCR comparativo e deteção de orientação)

`tesseract` não é um pacote de Python. Instalação por sistema:

```bash
# Debian, Ubuntu, Mint
sudo apt install tesseract-ocr tesseract-ocr-por

# Fedora
sudo dnf install tesseract tesseract-langpack-por

# Arch, Manjaro
sudo pacman -S tesseract tesseract-data-por

# macOS (Homebrew)
brew install tesseract tesseract-lang
```

No Windows não há gestor de pacotes de sistema fiável para isto. Usar o instalador do projeto
UB Mannheim, em https://github.com/UB-Mannheim/tesseract/wiki. Durante a instalação, abrir a
lista de línguas adicionais e marcar **Portuguese**, que instala o `por.traineddata` sem o qual
o Tesseract não lê os talões. Instalar na pasta por omissão, `C:\Program Files\Tesseract-OCR`:
é aí que o `pytesseract` procura o executável. Se for instalado noutro sítio, indicar o caminho
em `TESSERACT_CMD` no `.env`.

Confirmar a instalação, em qualquer sistema:

```bash
tesseract --version
tesseract --list-langs     # tem de listar "por"
```

O Tesseract também serve de sonda de orientação: `src.vision.preprocessing.detect_orientation_osd` chama o
binário em modo `--psm 0` para corrigir fotos de cabeça para baixo. Se o binário não existir, a
sonda devolve `0` e o pipeline segue apenas com a heurística de proporção, sem falhar.

#### zbar (leitura alternativa de QR)

O `pyzbar` é a primeira tentativa de descodificação do QR da AT e precisa da biblioteca `zbar`
do sistema:

```bash
# Debian, Ubuntu, Mint
sudo apt install libzbar0

# Fedora
sudo dnf install zbar

# Arch, Manjaro
sudo pacman -S zbar

# macOS (Homebrew)
brew install zbar
```

Sem ela, o `pyzbar` não importa e o `src.vision.qr_reader` recorre sozinho ao `cv2.QRCodeDetector`, que
lê os QR do golden set. Nenhum teste depende do `pyzbar`.

#### llama-server (LLM local)

Só é necessário para a extração por LLM local. A aplicação e os testes arrancam sem ele.

A via mais simples é descarregar um binário pré-compilado dos releases oficiais em
https://github.com/ggml-org/llama.cpp/releases. Escolher o build mais recente (as tags com o
formato `b12015`); a tag marcada como "latest" não traz binários. Depois escolher a variante
do sistema e da GPU:

| Sistema | Ficheiro a descarregar |
| :--- | :--- |
| Linux, AMD ou Intel | `llama-*-bin-ubuntu-vulkan-x64.tar.gz` |
| Linux, só CPU | `llama-*-bin-ubuntu-x64.tar.gz` |
| macOS, Apple Silicon | `llama-*-bin-macos-arm64.tar.gz` |
| macOS, Intel | `llama-*-bin-macos-x64.tar.gz` |
| Windows, AMD ou Intel | `llama-*-bin-win-vulkan-x64.zip` |
| Windows, só CPU | `llama-*-bin-win-cpu-x64.zip` |

Compilar a partir do código-fonte:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Linux com Vulkan (GPU integrada antiga ou dedicada)
cmake -B build -DGGML_VULKAN=ON

# macOS com Metal
cmake -B build -DGGML_METAL=ON

cmake --build build --config Release -j
```

No Windows, compilar exige o Vulkan SDK e as Visual Studio Build Tools; o binário
pré-compilado evita essa instalação. O executável chama-se `llama-server.exe` e tem de ser
chamado a partir da pasta onde foi descompactado, ou depois de essa pasta entrar no `PATH`. O
backend Vulkan, em qualquer sistema, precisa do runtime Vulkan que vem com o driver da gráfica.

No macOS, se o binário descarregado da Internet for bloqueado pelo Gatekeeper, remover a marca
de quarentena uma vez: `xattr -d com.apple.quarantine llama-server`.

Os pesos do modelo (`.gguf`) ficam em `models/` e não são versionados: o `.gitignore` exclui
`models/*.gguf` e `models/*.bin`.

## Instalação

Em Linux e macOS:

```bash
uv sync
cp .env.example .env      # opcional: o projeto arranca sem chaves de API
```

Em Windows (PowerShell):

```powershell
uv sync
Copy-Item .env.example .env
```

Em Windows (Prompt de Comando):

```bat
uv sync
copy .env.example .env
```

O `uv sync` instala também o grupo `dev` com `pytest`, `ruff` e `pyright`.

### Instalar o `uv`

Se ainda não estiver instalado:

```bash
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS, com Homebrew
brew install uv
```

```powershell
# Windows, PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Windows, com winget
winget install --id=astral-sh.uv -e
```

### Notas por sistema

- **Windows:** definir `$env:PYTHONUTF8 = "1"` na sessão antes de arrancar a aplicação. Sem
  isto a consola usa a code page do sistema e a impressão de acentos e símbolos falha com
  `UnicodeEncodeError`. Para o tornar permanente, `setx PYTHONUTF8 1`.
- **Windows:** os caminhos no código são construídos com `pathlib`. Não escrever caminhos
  literais como `r"C:\..."` em código novo, nem concatenar com `os.sep`.
- **Windows:** evitar manter o repositório dentro de pastas sincronizadas pelo OneDrive. A
  sincronização bloqueia ficheiros do `.venv` e produz erros de permissão difíceis de atribuir
  à causa real.
- **macOS:** o Homebrew instala em `/opt/homebrew/bin` nos Apple Silicon e em `/usr/local/bin`
  nos Intel; ambos já estão no `PATH` do Terminal nativo. Não usar o Terminal sob Rosetta.
- **Todos:** o `.gitattributes` normaliza as terminações de linha, para que um ficheiro
  editado no Windows não apareça como alterado em bloco no Linux e no macOS.

## Utilização

### Pré-visualizações por módulo

Além da interface Gradio, cada fase do pipeline tem uma entrada de linha de comando que abre uma
janela com o resultado sobre o primeiro talão de `data/receipts/` ou imprime o resultado na consola:

```bash
./run check-dataset                       # validação do golden set e estatísticas do corpus
uv run python -m src.vision.preprocessing # antes vs depois do OpenCV
uv run python -m src.ocr.easyocr          # linhas numeradas prontas para o prompt
uv run python -m src.ocr.tesseract        # OCR alternativo via Tesseract
uv run python -m src.vision.qr_reader     # campos do QR da AT, quando o talão tiver um
uv run python -m src.extraction.llm_client # teste de ligação ao cliente LLM
uv run python -m src.extraction.regex     # extração estruturada baseline
uv run python -m src.extraction.llm       # extração estruturada por LLM
uv run python -m src.vision.evidence      # caixas do OCR e destaque das linhas citadas
uv run python -m src.extraction.qa_assistant # pergunta de teste ao assistente
uv run python -m src.evaluation.evaluate  # runner do benchmark oficial sobre o golden set
```

Flags comuns: `--file caminho/para/talao.jpg` escolhe a imagem e `--save saida.png` escreve o
resultado em disco em vez de abrir a janela. O `src.vision.evidence` acrescenta `--lines 1,2,3` para
escolher as linhas a destacar e a recortar; sem a flag, usa até três linhas que contenham
`TOTAL`, `NIF`, `NIPC` ou `CONTRIBUINTE` e, se não houver nenhuma, as duas primeiras. O
`src.vision.preprocessing` aceita `--rotate`, `--no-crop`, `--no-shadows`, `--no-osd` e `--deskew` /
`--no-deskew`; o `src.ocr.easyocr` aceita `--gpu` e `--no-preprocess`; o `src.vision.qr_reader` aceita
`--no-preprocess`.

### Aplicação Gradio

```bash
./run app          # Linux / macOS (ou 'run app' no Windows)
# ou diretamente via uv:
uv run python -m src.app
```

Arranca o servidor Gradio local (por omissão em `http://127.0.0.1:7860`) e expõe três separadores:
1. **Leitor e Assistente de Documentos**: extração dos cinco campos estruturados com destaque visual e assistente de Q&A verificável.
2. **Pipeline Passo a Passo**: inspeção visual das quatro fases de Visão Computacional (bruta, alinhada/recortada, filtrada e caixas delimitadoras de OCR).
3. **Validação e Benchmark**: dashboard com resumo das métricas e relatório do golden set.

Se o extrator escolhido não estiver disponível (por exemplo, `LLM Local (llama.cpp)` sem o
`llama-server` a correr), a aplicação mostra um aviso no ecrã e recorre ao `Regex Baseline`
como alternativa. O aviso fica à cabeça da caixa de metadados e a coluna `Extracao` indica o
método que produziu de facto os valores. O assistente de Q&A não tem alternativa: sem
`llama-server` a resposta é um erro explícito com a instrução para arrancar `./run serve`.

Na primeira execução o EasyOCR descarrega os pesos do modelo para `~/.EasyOCR` (no Windows,
`C:\Users\<nome>\.EasyOCR`), o que exige rede. Correr uma vez com rede antes de qualquer
demonstração offline: a partir daí os pesos ficam em cache local. Isto aplica-se às
pré-visualizações de OCR e de evidência; a de pré-processamento só usa OpenCV.

Para o LLM local, arrancar primeiro o servidor num terminal separado, a partir da raiz do
repositório, com `llama-server` no `PATH`. O lançador usa por omissão o
`models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` e `-ngl 0` (seguro para CPU, que corre
confortavelmente o modelo 3B e evita erros de alocação de VRAM em placas pequenas):

```bash
./run serve        # Linux / macOS (ou 'run serve' no Windows)
```

Com uma GPU com pelo menos 3 GB livres, passar `-ngl 99`:

```bash
./run serve -ngl 99
```

O `models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` também corre no servidor como modelo de texto,
mas a entrada de imagens falha sem o ficheiro `mmproj` correspondente (`llama.cpp` precisa do
projetor multimodal para processar imagens).

O endereço por omissão em `.env.example` já aponta para `http://127.0.0.1:8080/v1`; se a porta
8080 estiver ocupada, mudar `PORT` e `LLAMA_BASE_URL` em conjunto.

## Testes

A suite é offline, determinística e não carrega modelos nem motores OCR reais. Corre em cerca
de 4,5 segundos e os comandos são idênticos em Linux, macOS e Windows, porque passam todos pelo
`uv`:

```bash
uv run pytest
```

O `pyproject.toml` deseleciona os testes marcados com `slow`, que são os únicos que carregam
pesos reais. Para correr o teste ao vivo do EasyOCR (exige os pesos em cache):

```bash
uv run pytest -m slow
```

Comandos úteis:

```bash
uv run pytest tests/test_evidence.py -q             # um ficheiro
uv run pytest -k "nif" -q                           # filtra por nome
uv run pytest --cov=src --cov-report=term-missing   # com cobertura
```

Ficheiros atuais da suite (304 testes offline): `test_smoke.py`, `test_settings.py`,
`test_normalizers.py`, `test_golden_set.py`, `test_preprocessing.py`, `test_ocr_contract.py`,
`test_ocr_tesseract.py`, `test_qr_reader.py`, `test_evidence.py`, `test_llm_client.py`,
`test_extractor_regex.py`, `test_extractor_llm.py`, `test_qa_assistant.py`, `test_scoring.py`,
`test_dataset_stats.py`, `test_evaluate.py` e `test_app.py`.

Qualidade de código:

```bash
uv run ruff format --line-length 120 src tests
uv run ruff check src tests
uv run pyright src tests
```

O `pyright` descarrega uma distribuição de Node na primeira execução, por isso precisa de rede
uma vez. Se a máquina não tiver rede, `uv run pytest` continua a funcionar: a suite não depende
do `pyright`.

## Estrutura

```text
src/
  core/             contratos de dados, configurações e normalizadores puros
  vision/           OpenCV: pré-processamento, leitura do QR AT e evidência visual
  ocr/              motores OCR (EasyOCR, Tesseract) e factory unificada
  extraction/       extração estruturada (regex, LLM local/cloud) e assistente Q&A
  evaluation/       validação do golden set, pontuação, estatísticas e benchmark runner
  app.py            aplicação interativa Gradio com 3 separadores
tests/              suite pytest com 304 testes, offline e determinística
data/               imagens do golden set e verdade de referência
results/            saídas do benchmark citadas no relatório
Docs/               planeamento e apresentação
models/             pesos .gguf do LLM local (não versionados)
run, run.bat        atalhos de execução multiplataforma (poethepoet)
```

O plano de implementação, com os pacotes de trabalho e os respetivos donos, está em
`Docs/implementation_plan.md`; as regras para agentes e colaboradores estão em
`AGENTS.md`.
