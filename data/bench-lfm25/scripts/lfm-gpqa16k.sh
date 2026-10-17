#!/bin/bash
# GPQA-diamond COMPLETO (198) budget reasoning 16K — noexp + exp01
set -u
MODEL="$HOME/models/LFM25-24B-A2B-QAD-Q4_0.gguf"
BIN="$HOME/work/AgrillaMoE/dist/linux/agrillamoe"
PORT=8090
mkdir -p "$HOME/logs"
run_config() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] config $name (16K, full 198) ==="
  pkill -9 -f "dist/linux/agrillam[o]e" 2>/dev/null
  sleep 3
  setsid nohup "$BIN" -m "$MODEL" --jinja -ngl 99 -c 81920 -ctk q8_0 -ctv q8_0 \
      --reasoning-budget 16384 --no-browser --port $PORT "$@" \
      > "$HOME/logs/srv16-$name.log" 2>&1 &
  local ok=0
  for i in $(seq 1 240); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "SERVER $name NON PARTITO"; tail -5 "$HOME/logs/srv16-$name.log"; return 1; fi
  echo "[$(date +%H:%M:%S)] server up ($name), eval inizio"
  evalscope eval \
    --model "LFM25-16k-$name" \
    --eval-type openai_api \
    --api-url "http://127.0.0.1:$PORT/v1" \
    --datasets gpqa_diamond \
    --generation-config "{\"temperature\":0,\"max_tokens\":32768,\"timeout\":7200,\"stream\":true}" \
    --eval-batch-size 4 \
    --work-dir "$HOME/gpqa16k-$name" \
    --use-cache "$HOME/gpqa16k-$name" --rerun-review \
    > "$HOME/logs/eval16-$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] eval $name fatto"
}
run_config noexp --no-moe-expansion
run_config exp01 --moe-experts 8 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39
pkill -9 -f "dist/linux/agrillam[o]e" 2>/dev/null
echo "=== GPQA16K COMPLETATO $(date) ==="
