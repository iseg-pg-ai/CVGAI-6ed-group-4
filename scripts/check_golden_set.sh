#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Validating Golden Set CSV against receipts ==="
uv run python -m src.evaluation.golden_set

echo ""
echo "=== 2. Computing Corpus Statistics ==="
uv run python -m src.evaluation.dataset_stats
