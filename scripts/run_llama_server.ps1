<#
.SYNOPSIS
    Starts llama-server on Windows PowerShell for receipt extraction and verifiable Q&A.
.DESCRIPTION
    Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.
#>
param (
    [string]$Model = "models\Qwen2.5-3B-Instruct-Q8_0.gguf",
    [int]$Port = 8080,
    [int]$Ngl = 99,
    [int]$Ctx = 8192
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Model)) {
    Write-Error "Error: Model file not found at $Model. Please place your .gguf model in models\ or specify: .\scripts\run_llama_server.ps1 -Model <path>"
    exit 1
}

Write-Host "Starting llama-server with model: $Model on port: $Port (NGL: $Ngl, CTX: $Ctx)"
& llama-server.exe -m $Model --port $Port -ngl $Ngl -c $Ctx
