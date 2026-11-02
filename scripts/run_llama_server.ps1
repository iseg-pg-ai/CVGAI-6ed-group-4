<#
.SYNOPSIS
    Starts llama-server on Windows PowerShell for receipt extraction and verifiable Q&A.
.DESCRIPTION
    Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.

    Configuration precedence:
      1. Script parameters: -Model, -Port, -Ngl, -Ctx
      2. Environment variables: MODEL, PORT, NGL, CTX
      3. .env file: LLAMA_MODEL_PATH / MODEL, LLAMA_PORT / PORT, etc.
      4. Default fallback: models\Qwen2.5-3B-Instruct-Q8_0.gguf, 8080, 99, 8192
.EXAMPLE
    .\scripts\run_llama_server.ps1
.EXAMPLE
    .\scripts\run_llama_server.ps1 -Model "models\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" -Ngl 0
#>
param (
    [string]$Model = "",
    [int]$Port = 0,
    [int]$Ngl = -1,
    [int]$Ctx = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$EnvFile = Join-Path $RepoRoot ".env"

# Load .env into process environment if present
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                $key = $parts[0].Trim()
                $val = $parts[1].Trim()
                if ($val -match '^["''](.*)["'']$') {
                    $val = $matches[1]
                } else {
                    $val = ($val -split '\s*#')[0].Trim()
                }
                if (-not [System.Environment]::GetEnvironmentVariable($key)) {
                    [System.Environment]::SetEnvironmentVariable($key, $val)
                }
            }
        }
    }
}

# Resolve settings by precedence
if (-not $Model) {
    $Model = if ($env:MODEL) { $env:MODEL } elseif ($env:LLAMA_MODEL_PATH) { $env:LLAMA_MODEL_PATH } else { "models\Qwen2.5-3B-Instruct-Q8_0.gguf" }
}

if ($Port -le 0) {
    $Port = if ($env:PORT) { [int]$env:PORT } elseif ($env:LLAMA_PORT) { [int]$env:LLAMA_PORT } else { 8080 }
}

if ($Ngl -lt 0) {
    $Ngl = if ($null -ne $env:NGL) { [int]$env:NGL } elseif ($null -ne $env:LLAMA_NGL) { [int]$env:LLAMA_NGL } else { 99 }
}

if ($Ctx -le 0) {
    $Ctx = if ($env:CTX) { [int]$env:CTX } elseif ($env:LLAMA_CTX) { [int]$env:LLAMA_CTX } else { 8192 }
}

if (-not (Test-Path $Model)) {
    Write-Error "Error: Model file not found at $Model. Please configure MODEL in .env, place your .gguf model in models\, or specify: .\scripts\run_llama_server.ps1 -Model <path>"
    exit 1
}

Write-Host "Starting llama-server with model: $Model on port: $Port (NGL: $Ngl, CTX: $Ctx)"
& llama-server.exe -m $Model --port $Port -ngl $Ngl -c $Ctx
