#!/usr/bin/env bash
# Scarica il fork llama.cpp (branch moe-expansion) come sorgente per AgrillaMoE.
# Serve solo se non esiste gia' ../repo (layout di sviluppo) — la build cerca,
# in ordine: -DAGRILLA_LLAMA_DIR, ../repo, ./llama.cpp
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$HERE/../repo/tools/server" ]; then
    echo "Trovato ../repo (fork moe-expansion): nessun clone necessario."
    exit 0
fi
if [ -d "$HERE/llama.cpp/tools/server" ]; then
    echo "Trovato $HERE/llama.cpp: nessun clone necessario."
    exit 0
fi
git clone -b moe-expansion https://github.com/vagrillo/llama.cpp "$HERE/llama.cpp"
echo "OK: fork clonato in $HERE/llama.cpp"
