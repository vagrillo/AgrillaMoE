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

# Il modello MOExp MixedQuant richiede il fork con pool per-esperto e le fix
# Metal per-pair: un clone vecchio o upstream compila ma poi fallisce al load
# ('ffn_down_exps.weight' not found) o in prefill. Verifica marker minimi.
if ! grep -q "FFN_EXPS_LOCAL_0" "$LLAMA_DIR/src/llama-arch.h" 2>/dev/null; then
    echo "ERRORE: $LLAMA_DIR non contiene il supporto pool (vecchio o upstream)."
    echo "  Aggiorna il fork e ripeti, es.:"
    echo "    git -C $LLAMA_DIR fetch origin moe-expansion && git -C $LLAMA_DIR reset --hard origin/moe-expansion"
    echo "  oppure clona: git clone -b moe-expansion https://github.com/vagrillo/llama.cpp \"$LLAMA_DIR\""
    exit 1
fi
if ! grep -q "nmatch" "$LLAMA_DIR/ggml/src/ggml-metal/kernels/mul_mm.metal" 2>/dev/null; then
    echo "WARNING: $LLAMA_DIR non contiene la fix Metal per-pair (commit 22fb86823):"
    echo "  il prefill con il modello mixed puo' crashare/corrompere l'output."
    echo "  Aggiorna: git -C $LLAMA_DIR fetch origin moe-expansion && git -C $LLAMA_DIR reset --hard origin/moe-expansion"
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
