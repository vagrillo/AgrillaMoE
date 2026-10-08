#!/bin/bash
# run-expbench.sh — benchmark qualità MoE-expansion: 5 problemi × 10 config
#
# Uso su una VM GPU 16GB (es. V100) con quantizzazione UD-Q3_K_XL:
#   bash setup-expbench.sh && bash run-expbench.sh
#
# Variabili:
#   EXPBENCH_MODEL   modello GGUF (default models/Q3KXL.gguf)
#   EXPBENCH_CM=1    esperti su CPU -cmoe (necessario su 16GB per far stare
#                    KV 32K + thinking 24K; per GPU 24GB+ lasciare unset)
#   EXPBENCH_BUDGET  reasoning budget (default 24576)
#   EXPBENCH_CTX     contesto (default 32768)
set -uo pipefail
cd "$(dirname "$0")"
export EXPBENCH_MODEL="${EXPBENCH_MODEL:-models/Q3KXL.gguf}"
export EXPBENCH_CM="${EXPBENCH_CM:-}"
export EXPBENCH_BUDGET="${EXPBENCH_BUDGET:-24576}"
export EXPBENCH_CTX="${EXPBENCH_CTX:-32768}"

echo "=== STAGE 1: generazione 10 config x 5 problemi (resumabile) ==="
python3 expbench.py --stage gen --model "$EXPBENCH_MODEL"

echo "=== STAGE 2: test oggettivi ==="
python3 expbench.py --stage tests

echo "=== STAGE 3: judge LLM accecato (server stock) ==="
python3 expbench.py --stage judge --model "$EXPBENCH_MODEL"

echo
echo "Risultati: runs/judge-report.json, runs/tests-summary.json, runs/<cfg>/p*.json"
