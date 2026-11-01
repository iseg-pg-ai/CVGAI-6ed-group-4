#!/usr/bin/env bash
# Start llama-server for local receipt extraction and verifiable Q&A.
# Ref: Docs/Planning/implementation_plan.md, section 1.1 and WP-08.
set -euo pipefail

MODEL="${1:-models/Qwen2.5-3B-Instruct-Q8_0.gguf}"
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX="${CTX:-8192}"

if [ ! -f "${MODEL}" ]; then
    echo "Error: Model file not found at ${MODEL}"
    echo "Please place your .gguf model in models/ or specify path: $0 <path/to/model.gguf>"
    exit 1
fi

echo "Starting llama-server with model: ${MODEL} on port: ${PORT} (NGL: ${NGL}, CTX: ${CTX})"
exec llama-server -m "${MODEL}" --port "${PORT}" -ngl "${NGL}" -c "${CTX}"
