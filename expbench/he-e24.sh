#!/bin/bash
# he-e24.sh — HumanEval run singolo con espansione e24 (24 esperti / 0.8 / L25-39)
#
# Stesse modalità esatte del precedente A/B (stock 89.63%, e20 90.85%):
#   modello IQ4_XS 17.7GB full GPU, budget reasoning 4096, ctx 16384, KV q8_0,
#   temperatura 0, 164 problemi, un campione.
# Il confronto è con i risultati storici salvati in data-moe/expbench/
#   (he-noexp.json 89.63 / he-e20.json 90.85).
#
# Uso su VM GPU (24GB+ full GPU; su 16-20GB aggiungere -cmoe):
#   bash expbench/he-e24.sh [modello.gguf]
set -uo pipefail
cd "$(dirname "$0")/.."   # radice AgrillaMoE

M="${1:-models/IQ4XS.gguf}"
PORT=8097
BUDGET="${HE_BUDGET:-4096}"
NPROB="${HE_NPROB:-164}"

[ -f "$M" ] || { echo "modello mancante: $M"; echo "scarica: curl -sL --fail -o $M https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"; exit 1; }
[ -f dist/linux/agrillamoe ] || { echo "build mancante: AGRILLA_JOBS=\$(nproc) ./build-linux.sh"; exit 1; }
[ -f humaneval_run.py ] || { echo "humaneval_run.py mancante"; exit 1; }
pip3 install -q datasets 2>&1 | tail -1 || true

wait_up() {
  for i in $(seq 1 300); do
    sleep 3
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 $1 2>/dev/null || return 1
  done
  return 1
}

LOG=/root/he-e24-server.log
rm -f "$LOG" /root/he-e24.json
echo "###### HUMANEVAL e24 (24 esperti / 0.8 / L25-39) — budget $BUDGET — $(date) ######"
./dist/linux/agrillamoe -m "$M" \
    --moe-experts 24 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39 \
    --flash-attn on -ctk q8_0 -ctv q8_0 --fit off -ngl 99 \
    --host 127.0.0.1 --port $PORT --no-browser \
    -c 16384 -np 1 > "$LOG" 2>&1 &
SRV=$!
wait_up $SRV || { echo "SERVER NON PRONTO"; tail -8 "$LOG"; kill $SRV; exit 1; }
echo "server pronto (banner espansione: $(grep -m1 -oE 'esperti [0-9]+, soglia [0-9.]+' "$LOG" || echo 'vedi log'))"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
python3 humaneval_run.py $PORT /root/he-e24.json $NPROB 6144
echo "--- t/s medie dal log ---"
grep "print_timing" "$LOG" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | \
  awk -F" " "{s+=\$1; n++} END {if (n) printf \"decode media: %.1f t/s su %d richieste\n\", s/n, n}"
kill $SRV 2>/dev/null
echo "###### FINE — $(date) — confronto con: no-exp 89.63 (69.3 t/s) | e20 90.85 (56.0 t/s) ######"
