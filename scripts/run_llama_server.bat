@echo off
rem Starts llama-server on Windows Command Prompt for receipt extraction and Q&A.
rem Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.
setlocal

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=models\Qwen2.5-3B-Instruct-Q8_0.gguf"
if "%PORT%"=="" set "PORT=8080"
if "%NGL%"=="" set "NGL=99"
if "%CTX%"=="" set "CTX=8192"

if not exist "%MODEL%" (
    echo Error: Model file not found at %MODEL%
    echo Please place your .gguf model in models\ or specify: %0 ^<path\to\model.gguf^>
    exit /b 1
)

echo Starting llama-server with model: %MODEL% on port: %PORT% (NGL: %NGL%, CTX: %CTX%)
llama-server.exe -m "%MODEL%" --port %PORT% -ngl %NGL% -c %CTX%
