@echo off
rem Starts llama-server on Windows Command Prompt for receipt extraction and Q&A.
rem Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.
rem
rem Configuration precedence:
rem   1. CLI argument:       run_llama_server.bat <path\to\model.gguf>
rem   2. Environment:        MODEL, PORT, NGL, CTX
rem   3. .env file:          LLAMA_MODEL_PATH / MODEL, LLAMA_PORT / PORT, etc.
rem   4. Default fallback:   models\Qwen2.5-3B-Instruct-Q8_0.gguf, 8080, 99, 8192
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%..\.env"

rem Load .env if present and variables not already defined
if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1* delims==" %%A in ("%ENV_FILE%") do (
        set "KEY=%%A"
        set "VAL=%%B"
        for /f "tokens=* delims= " %%K in ("!KEY!") do set "KEY=%%K"
        for /f "tokens=* delims= " %%V in ("!VAL!") do set "VAL=%%V"
        if not "!KEY!"=="" (
            for /f "tokens=1 delims=#" %%C in ("!VAL!") do set "VAL=%%C"
            for /l %%I in (1,1,31) do if "!VAL:~-1!"==" " set "VAL=!VAL:~0,-1!"
            set "VAL=!VAL:"=!"
            set "VAL=!VAL:'=!"
            if not defined !KEY! set "!KEY!=!VAL!"
        )
    )
)

set "MODEL_ARG=%~1"
if not "%MODEL_ARG%"=="" set "MODEL=%MODEL_ARG%"
if not defined MODEL if defined LLAMA_MODEL_PATH set "MODEL=%LLAMA_MODEL_PATH%"
if not defined MODEL set "MODEL=models\Qwen2.5-3B-Instruct-Q8_0.gguf"

if not defined PORT if defined LLAMA_PORT set "PORT=%LLAMA_PORT%"
if not defined PORT set "PORT=8080"

if not defined NGL if defined LLAMA_NGL set "NGL=%LLAMA_NGL%"
if not defined NGL set "NGL=99"

if not defined CTX if defined LLAMA_CTX set "CTX=%LLAMA_CTX%"
if not defined CTX set "CTX=8192"

if not exist "%MODEL%" (
    echo Error: Model file not found at %MODEL%
    echo Please configure MODEL in .env, place your .gguf model in models\ or specify: %0 ^<path\to\model.gguf^>
    exit /b 1
)

echo Starting llama-server with model: %MODEL% on port: %PORT% (NGL: %NGL%, CTX: %CTX%)
llama-server.exe -m "%MODEL%" --port %PORT% -ngl %NGL% -c %CTX%
