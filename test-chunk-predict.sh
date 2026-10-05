#!/bin/bash
# test-chunk-predict.sh — A/B sul 35B Q8_0 in gpu-streaming: baseline vs chunked predict
# Uso sulla VM: bash test-chunk-predict.sh [percorso-modello]
# Requisiti: repo AgrillaMoE clonato con build fatta (o MOE_FORCE_BUILD=1)
set -uo pipefail
cd "$(dirname "$0")"

M="${1:-models/Q8.gguf}"
if [ ! -f "$M" ]; then
    echo "scarico Q8_0 (34GB)..."
    mkdir -p models
    curl -sL --fail -o "$M" "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q8_0.gguf"
fi
[ -x dist/linux/agrillamoe ] || { echo "build..."; AGRILLA_JOBS=$(nproc) ./build-linux.sh; }

PORT=8095
run_case() {
  local NAME="$1"; shift
  local ENVS="$1"; shift
  local LOG="/root/ab-$NAME.log"
  rm -f "$LOG"
  echo "=== $NAME ==="
  env $ENVS ./dist/linux/agrillamoe -m "$M" $@ \
      --host 127.0.0.1 --port $PORT --no-browser \
      --reasoning off -c 2048 -np 1 > "$LOG" 2>&1 &
  local SRV=$!
  local OK=0
  for i in $(seq 1 250); do
    sleep 3
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { OK=1; break; }
    kill -0 $SRV 2>/dev/null || break
  done
  [ "$OK" != "1" ] && { echo "NON PRONTO"; tail -5 "$LOG"; kill $SRV 2>/dev/null; return; }
  # warmup lungo (le statistiche di transizione maturano nei primi token)
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Parla dell oceano."}],"max_tokens":130,"temperature":0}' >/dev/null
  # misura
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Parla della montagna."}],"max_tokens":200,"temperature":0}' >/dev/null
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  grep -E "eval time =" "$LOG" | tail -1
  grep -E "chunked predict attivo|moe chunked" "$LOG" | head -1
  kill $SRV 2>/dev/null
  sleep 5
}

echo "###### A/B chunked predict — $(date) ######"
run_case baseline  ""
run_case chunk     "LLAMA_MOE_CHUNK_AFTER=26,29,32,35,38 LLAMA_MOE_PREDICT_BUDGET=24" --agrilla-chunk-predict
echo "###### FINE — $(date) ######"
