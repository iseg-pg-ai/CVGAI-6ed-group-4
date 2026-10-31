# CVGAI 6ª Edição — Grupo 4

Leitor inteligente de talões e faturas simplificadas portuguesas. O sistema recebe a
fotografia de um documento (plano ou amarrotado), aplica pré-processamento com OpenCV,
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
(baseline), LLM local servido por `llama.cpp` e LLM na cloud (Gemini ou OpenAI).

## Requisitos

- Linux, macOS ou Windows. As três plataformas estão suportadas: o pipeline não usa
  binários específicos de um sistema e todos os caminhos são construídos com `pathlib`.
- `uv` (gestor de pacotes e de versões de Python).
- Python 3.12, obtido automaticamente pelo `uv` a partir de `.python-version`.
- `tesseract-ocr` com o pacote de língua portuguesa, apenas para o motor OCR comparativo.
- `llama-server`, apenas para a extração por LLM local.

### Dependências de sistema

O `uv` resolve o Python e os pacotes, mas o `tesseract` e o `llama-server` são binários do
sistema e instalam-se à parte. Nenhum dos dois é necessário para correr a suite de testes.

#### Tesseract (motor OCR comparativo)

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

Os pesos do modelo (`.gguf`) ficam em `models/`, que está no `.gitignore`.

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

```bash
uv run python -m src.app
```

A aplicação abre no browser e expõe três separadores: extração de campos, inspeção das
etapas de pré-processamento e perguntas sobre o documento.

Na primeira execução o EasyOCR descarrega os pesos do modelo para `~/.EasyOCR` (no Windows,
`C:\Users\<nome>\.EasyOCR`), o que exige rede. Correr a aplicação uma vez com rede antes de
qualquer demonstração offline: a partir daí os pesos ficam em cache local.

Para o LLM local, arrancar primeiro o servidor num terminal separado, a partir da raiz do
repositório, com o binário no `PATH`. O `-ngl 99` descarrega as camadas para a GPU; usar
`-ngl 0` para correr só em CPU.

```bash
llama-server -m models/modelo.gguf --port 8080 -ngl 99 -c 8192
```

Em Windows o comando é o mesmo, com `llama-server.exe`:

```powershell
llama-server.exe -m models\modelo.gguf --port 8080 -ngl 99 -c 8192
```

O endereço por omissão em `.env.example` já aponta para `http://127.0.0.1:8080/v1`; se a porta
8080 estiver ocupada, mudar `--port` e `LLAMA_BASE_URL` em conjunto.

## Testes

A suite é offline, determinística e não carrega modelos nem motores OCR reais. Corre em
segundos e os comandos são idênticos em Linux, macOS e Windows, porque passam todos pelo `uv`:

```bash
uv run pytest
```

Comandos úteis:

```bash
uv run pytest tests/test_normalizers.py -q          # um ficheiro
uv run pytest -k "nif" -q                           # filtra por nome
uv run pytest --cov=src --cov-report=term-missing   # com cobertura
```

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
src/            pipeline (pré-processamento, OCR, extração, Q&A, interface)
tests/          suite pytest
data/           imagens do golden set e verdade de referência
results/        saídas do benchmark citadas no relatório
Docs/           planeamento e apresentação
```