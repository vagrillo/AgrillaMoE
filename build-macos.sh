#!/usr/bin/env bash
# AgrillaMoE — build macOS (Apple Silicon, backend Metal incorporato)
#
# Su macOS il backend e' Metal (GGML_METAL, default su Apple): la MoE-expansion
# funziona identica (routing nel grafo, mul_mat_id supportato da Metal).
# La memoria unificata rende superflua la modalita' streaming: il "budget" e'
# la RAM totale (una M-series da 24-48 GB carica anche Q8_0 per intero).
#
# Variabili opzionali:
#   AGRILLA_BUILD_DIR  cartella di build (default: ~/agrilla-build-macos)
#   AGRILLA_JOBS       job paralleli (default: sysctl -n hw.ncpu)
#   AGRILLA_LLAMA_DIR  fork llama.cpp moe-expansion (default: ./llama.cpp o ../repo)
#
# Requisiti: Xcode Command Line Tools (xcode-select --install) e cmake (brew install cmake)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${AGRILLA_BUILD_DIR:-$HOME/agrilla-build-macos}"
JOBS="${AGRILLA_JOBS:-$(sysctl -n hw.ncpu)}"

LLAMA_DIR="${AGRILLA_LLAMA_DIR:-}"
if [ -z "$LLAMA_DIR" ]; then
    if [ -d "$HERE/llama.cpp/tools/server" ]; then LLAMA_DIR="$HERE/llama.cpp"
    elif [ -d "$HERE/../repo/tools/server" ]; then LLAMA_DIR="$HERE/../repo"
    else
        echo "fork moe-expansion non trovato: eseguo ./bootstrap-llama.sh"
        bash "$HERE/bootstrap-llama.sh"
        LLAMA_DIR="$HERE/llama.cpp"
    fi
fi

cmake -S "$HERE" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAGRILLA_LLAMA_DIR="$LLAMA_DIR"

cmake --build "$BUILD_DIR" --target agrillamoe -j"$JOBS"

BIN="$BUILD_DIR/agrillamoe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/bin/agrillamoe"
mkdir -p "$HERE/dist/macos"
cp -f "$BIN" "$HERE/dist/macos/agrillamoe"
echo
echo "OK: $HERE/dist/macos/agrillamoe ($(du -h "$HERE/dist/macos/agrillamoe" | cut -f1))"
