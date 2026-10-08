#!/bin/bash
# he-3way.sh — HumanEval A/B/C: stock vs e24 vs e20 (stesso reasoning budget)
#
# Uso su una VM GPU (consigliata ≥24GB per full GPU, oppure -cmoe su 16-20GB):
#   git clone https://github.com/vagrillo/AgrillaMoE && cd AgrillaMoE
#   git clone --depth 1 -b moe-expansion https://github.com/vagrillo/llama.cpp llama.cpp
#   bash expbench/he-3way.sh
#
# Requisiti: build fatto (./dist/linux/agrillamoe), humaneval_run.py, dataset
# HumanEval scaricato dal harness automaticamente.
set -uo pipefail
cd "$(dirname "$0")"

M="${HE_MODEL:-models/IQ4XS.gguf}"
PORT=8097
BUDGET="${HE_BUDGET:-4096}"
NPROB="${HE_NPROB:-164}"

if [ ! -f humaneval_run.py ]; then
    echo "humaneval_run.py mancante: copialo dalla radice del repo"; exit 1
fi
pip3 install -q datasets 2>&1 | tail -1 || true

wait_up() {
  for i in $(seq 1 300); do
    sleep 3
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 $1 2>/dev/null || return 1
  done
  return 1
}

run_config() {
  local NAME="$1"; shift
  local LOG="/root/he-$NAME-server.log"
  rm -f "$LOG" "/root/he-$NAME.json"
  echo "###### $NAME ######"
  ./dist/linux/agrillamoe -m "$M" "$@" \
      --flash-attn on -ctk q8_0 -ctv q8_0 --fit off -ngl 99 \
      --host 127.0.0.1 --port $PORT --no-browser \
      -c 16384 -np 1 > "$LOG" 2>&1 &
  local SRV=$!
  wait_up $SRV || { echo "SERVER NON PRONTO"; tail -8 "$LOG"; kill $SRV; return; }
  echo "server pronto"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  python3 humaneval_run.py $PORT "/root/he-$NAME.json" $NPROB 6144
  echo "--- t/s medie dal log ---"
  grep "print_timing" "$LOG" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | \
    awk -F" " "{s+=\$1; n++} END {if (n) printf \"decode media: %.1f t/s su %d richieste\n\", s/n, n}"
  kill $SRV 2>/dev/null
  sleep 5
}

echo "###### HUMANEVAL 3-way budget $BUDGET — $(date) ######"
run_config noexp --no-moe-expansion --reasoning-budget $BUDGET
run_config e24   --moe-experts 24 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39 --reasoning-budget $BUDGET
run_config e20   --moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39 --reasoning-budget $BUDGET
echo "###### FINE — $(date) ######"
echo "Risultati: /root/he-noexp.json /root/he-e24.json /root/he-e20.json"
