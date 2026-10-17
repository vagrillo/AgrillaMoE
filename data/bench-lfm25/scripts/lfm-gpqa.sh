#!/bin/bash
# GPQA-diamond (limit 50) LFM2.5-24B-A2B QAD-Q4_0 — noexp / exp01 / exp02
# reasoning budget 8K, porta 8090
set -u
MODEL="$HOME/models/LFM25-24B-A2B-QAD-Q4_0.gguf"
BIN="$HOME/work/AgrillaMoE/dist/linux/agrillamoe"
PORT=8090
LIMIT=50
mkdir -p "$HOME/logs"
run_config() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] config $name ==="
  pkill -f agrilla-build-linux/agrillamoe 2>/dev/null
  sleep 3
  setsid nohup "$BIN" -m "$MODEL" --jinja -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 \
      --reasoning-budget 8192 --no-browser --port $PORT "$@" \
      > "$HOME/logs/srv-$name.log" 2>&1 &
  local ok=0
  for i in $(seq 1 180); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { ok=1; break; }
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "SERVER $name NON PARTITO"; tail -5 "$HOME/logs/srv-$name.log"; return 1; fi
  echo "[$(date +%H:%M:%S)] server up (config $name), eval inizio"
  evalscope eval \
    --model "LFM25-$name" \
    --eval-type openai_api \
    --api-url "http://127.0.0.1:$PORT/v1" \
    --datasets gpqa_diamond \
    --limit $LIMIT \
    --generation-config "{\"temperature\":0,\"max_tokens\":16384,\"timeout\":3600,\"stream\":true}" \
    --eval-batch-size 4 \
    --work-dir "$HOME/gpqa-$name" \
    --use-cache "$HOME/gpqa-$name" --rerun-review \
    > "$HOME/logs/eval-$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] eval $name fatto: $(grep -A6 accuracy "$HOME/logs/eval-$name.log" | tail -3 | tr "\n" " ")"
}
run_config noexp --no-moe-expansion
run_config exp01 --moe-experts 8 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39
run_config exp02 --moe-experts 12 --moe-expert-threshold 0.8 --moe-expert-layer-start 20 --moe-expert-layer-end 39
pkill -f agrilla-build-linux/agrillamoe 2>/dev/null
echo "=== BENCHMARK COMPLETATO $(date) ==="
