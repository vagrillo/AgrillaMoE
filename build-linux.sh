#!/usr/bin/env bash
# AgrillaMoE — build Linux (staticamente linkato, backend CUDA incorporato)
#
# Variabili d'ambiente opzionali:
#   AGRILLA_CUDA_ARCH  architetture CUDA (default: native; es. "61", "70;80;86")
#   AGRILLA_BUILD_DIR  cartella di build (default: ~/agrilla-build-linux)
#   AGRILLA_JOBS       job paralleli (default: 6)
#   AGRILLA_NATIVE     1 (default) = -march=native del build host;
#                      0 = CPU baseline portabile (consigliato per binari
#                      da ridistribuire su macchine diverse)
#
# Binario portabile da distribuire (esempio):
#   AGRILLA_NATIVE=0 AGRILLA_CUDA_ARCH="61;70;80;86" ./build-linux.sh
#
# Requisiti: cmake >= 3.24, gcc/g++, CUDA toolkit 12+ (nvcc in PATH o /usr/local/cuda)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${AGRILLA_BUILD_DIR:-$HOME/agrilla-build-linux}"
ARCH="${AGRILLA_CUDA_ARCH:-native}"
JOBS="${AGRILLA_JOBS:-6}"
NATIVE="${AGRILLA_NATIVE:-1}"

# nvcc: preferisci il toolkit NVIDIA reale (es. /usr/local/cuda) al pacchetto
# distro (spesso una versione CUDA vecchia, incompatibile con gcc recenti)
NVCC_BIN="$(command -v nvcc || true)"
for CAND in /usr/local/cuda/bin/nvcc /usr/local/cuda-12*/bin/nvcc; do
    if [ -x "$CAND" ]; then NVCC_BIN="$CAND"; break; fi
done
if [ -z "$NVCC_BIN" ]; then
    echo "ERRORE: nvcc non trovato (serve CUDA toolkit 12+)" >&2
    exit 1
fi
echo "uso nvcc: $NVCC_BIN ($("$NVCC_BIN" --version | tail -1))"

cmake -S "$HERE" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE="$NATIVE" \
    -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

cmake --build "$BUILD_DIR" --target agrillamoe -j"$JOBS"

mkdir -p "$HERE/dist/linux"
BIN="$BUILD_DIR/agrillamoe"
[ -f "$BIN" ] || BIN="$BUILD_DIR/bin/agrillamoe"
cp -f "$BIN" "$HERE/dist/linux/agrillamoe"
echo
echo "OK: $HERE/dist/linux/agrillamoe ($(du -h "$HERE/dist/linux/agrillamoe" | cut -f1))"
