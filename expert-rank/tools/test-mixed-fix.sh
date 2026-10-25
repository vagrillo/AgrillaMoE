#!/usr/bin/env bash
# Test funzionale post-fix su V100: copre tutti i percorsi mul_mat_id
#   - ubatch <= 8 token   -> MMVQ
#   - ubatch  9..64 token -> MMQ + scatter quantize (per-pair mm_ids_helper)
#   - ubatch > 64 token   -> fallback sorted (compazione per-pair host-side)
# Verifica che il server sopravviva e che le risposte greedy siano coerenti.
set -u
PORT=9080
LOG=/workspace/test-fix.log
pkill -9 -f llama-server 2>/dev/null; sleep 2

setsid nohup /workspace/llama.cpp/build/bin/llama-server \
  -m /tmp/models/Qwen3.6-35B-A3B-mixed-expert.gguf \
  -ngl 999 -c 8192 -ctk q8_0 -ctv q8_0 \
  --temp 0 --jinja --parallel 1 \
  --moe-experts 20 --moe-expert-threshold 0.95 \
  --host 0.0.0.0 --port $PORT > "$LOG" 2>&1 < /dev/null &

t0=$SECONDS
while ! curl -fsS http://127.0.0.1:$PORT/health >/dev/null 2>&1; do
  kill -0 %1 2>/dev/null || { echo "SERVER MORTO in avvio"; tail -5 "$LOG"; exit 1; }
  [ $((SECONDS - t0)) -gt 600 ] && { echo "timeout health"; exit 1; }
  sleep 2
done
echo "server pronto in $((SECONDS-t0))s"

gen_prompt() {  # $1 = parole circa
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
text = " ".join(out)[:n*11]
print(text)
EOF
}

ask() {  # $1 label, $2 prompt, $3 n_predict
  local label="$1" prompt="$2" npred="$3"
  local body resp
  body=$(python3 - "$prompt" "$npred" <<'EOF'
import json, sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]),
                  "temperature": 0, "cache_prompt": False}))
EOF
)
  resp=$(curl -fsS --max-time 600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d "$body") || {
    echo "[$label] REQUEST FAILED"; tail -3 "$LOG"; exit 1; }
  local txt pn
  txt=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d['content'])[:220]); print('prompt_n=', d['timings']['prompt_n'], 'predicted_n=', d['timings']['predicted_n'])" <<<"$resp")
  echo "[$label] $txt"
}

P_TINY=$(gen_prompt 6)      # ~8 token
P_SHORT=$(gen_prompt 30)    # ~40 token
P_MID=$(gen_prompt 220)     # ~300 token
P_LONG=$(gen_prompt 520)    # ~700 token (2 ubatch: 512 + resto)

echo "== coarse sanity: testo generato coerente su tutti i percorsi =="
ask tiny  "$P_TINY"  24
ask short "$P_SHORT" 24
ask mid   "$P_MID"   24
ask long  "$P_LONG"  24

echo "== determinismo greedy (stesso prompt due volte) =="
ask det1  "$P_SHORT" 24
ask det2  "$P_SHORT" 24

echo "== lunga generazione (decode esteso su MMVQ) =="
ask gen   "$P_SHORT" 96

if grep -q "GGML_ASSERT\|CUDA error\|unspecified launch" "$LOG"; then
  echo "FAIL: errori nel log server"; grep -m3 "GGML_ASSERT\|CUDA error" "$LOG"; exit 1
fi
echo "PASS: nessun crash, tutti i percorsi eseguiti"
