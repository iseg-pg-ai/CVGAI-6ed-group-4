#!/usr/bin/env bash
# Start llama-server for local receipt extraction and verifiable Q&A.
# Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.
#
# Configuration precedence:
#   1. CLI argument:       $0 <path/to/model.gguf>
#   2. Environment:        MODEL=... PORT=... NGL=... CTX=...
#   3. .env file:          LLAMA_MODEL_PATH / MODEL, LLAMA_PORT / PORT, etc.
#   4. Default fallback:   models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf, 8080, 0, 8192
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# Load .env if present and variables are not already set in the environment
if [ -f "${ENV_FILE}" ]; then
    while IFS= read -r line || [ -n "${line}" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        case "${line}" in
            ""|\#*) continue ;;
        esac
        if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            val="${val%%#*}"
            val="${val%"${val##*[![:space:]]}"}"
            val="${val#\"}"
            val="${val%\"}"
            val="${val#\'}"
            val="${val%\'}"
            if [ -z "${!key+x}" ]; then
                export "${key}=${val}"
            fi
        fi
    done < "${ENV_FILE}"
fi

MODEL="${1:-${MODEL:-${LLAMA_MODEL_PATH:-models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf}}}"
PORT="${PORT:-${LLAMA_PORT:-8080}}"
NGL="${NGL:-${LLAMA_NGL:-0}}"
CTX="${CTX:-${LLAMA_CTX:-8192}}"

if [ ! -f "${MODEL}" ]; then
    echo "Error: Model file not found at ${MODEL}"
    echo "Please configure MODEL in .env, place your .gguf model in models/, or specify path: $0 <path/to/model.gguf>"
    exit 1
fi

echo "Starting llama-server with model: ${MODEL} on port: ${PORT} (NGL: ${NGL}, CTX: ${CTX})"
exec llama-server -m "${MODEL}" --port "${PORT}" -ngl "${NGL}" -c "${CTX}"
