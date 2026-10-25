#!/usr/bin/env bash
# Memcheck dei percorsi mul_mat_id del modello mixed-pool:
#   tiny  (~8 tok)  -> MMVQ
#   short (~40 tok) -> MMQ + scatter quantize per-pair
#   mid   (~100 tok)-> fallback sorted per-pair (> MMQ_DP4A_MAX_BATCH_SIZE)
# PASS solo se compute-sanitizer riporta ERROR SUMMARY: 0 errors.
set -u
PORT=9083
LOG=/workspace/san-fix.log
pkill -9 -f llama-server 2>/dev/null; sleep 2

setsid nohup /usr/local/cuda/bin/compute-sanitizer --tool memcheck --launch-timeout 240 \
  /workspace/llama.cpp/build/bin/llama-server \
  -m /tmp/models/Qwen3.6-35B-A3B-mixed-expert.gguf \
  -ngl 999 -c 4096 -ctk q8_0 -ctv q8_0 \
  --temp 0 --jinja --parallel 1 \
  --moe-experts 20 --moe-expert-threshold 0.95 \
  --host 0.0.0.0 --port $PORT > "$LOG" 2>&1 < /dev/null &

t0=$SECONDS
while ! curl -fsS http://127.0.0.1:$PORT/health >/dev/null 2>&1; do
  kill -0 %1 2>/dev/null || { echo "SERVER MORTO in avvio"; tail -5 "$LOG"; exit 1; }
  [ $((SECONDS - t0)) -gt 900 ] && { echo "timeout health"; exit 1; }
  sleep 3
done
echo "server (sanitizer) pronto in $((SECONDS-t0))s"

gen_prompt() {
  python3 - "$1" <<'EOF'
import sys
n = int(sys.argv[1])
words = ("analysis of quantization error in mixture of experts models shows "
         "that per expert mixed precision can preserve quality while reducing "
         "memory the router selects the top experts and the remaining slots "
         "are masked with zero weight so the contribution vanishes exactly "
         "as in the unpooled compute this property holds for every layer")
out = []
while sum(len(w.split()) for w in out) < n:
    out.append(words)
print(" ".join(out)[:n*11])
EOF
}

ask() {
  local label="$1" prompt="$2"
  local body resp
  body=$(python3 - "$prompt" <<'EOF'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": 8, "temperature": 0, "cache_prompt": False}))
EOF
)
  curl -fsS --max-time 3600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d "$body" >/dev/null \
    || { echo "[$label] REQUEST FAILED"; tail -3 "$LOG"; exit 1; }
  echo "[$label] ok"
}

ask tiny  "$(gen_prompt 6)"
ask short "$(gen_prompt 30)"
ask mid   "$(gen_prompt 75)"

pkill -9 -f llama-server 2>/dev/null; sleep 3

if grep -q "Invalid __global__\|Invalid __shared__\|out of bounds\|misaligned" "$LOG"; then
  echo "FAIL: sanitizer ha trovato errori"; grep -m3 -B1 -A3 "Invalid\|out of bounds" "$LOG"; exit 1
fi
if ! grep -q "ERROR SUMMARY: 0 errors" "$LOG"; then
  echo "FAIL: ERROR SUMMARY != 0"; grep -m3 "ERROR SUMMARY" "$LOG"; exit 1
fi
echo "PASS: memcheck pulito su tutti i percorsi (MMVQ/MMQ-scatter/fallback)"
