#!/usr/bin/env bash
# moe-predict-run.sh — sessione lunga di query + log esperti + predictions
#
# Pensato per una VM GPU di vast.ai (ma gira ovunque): costruisce AgrillaMoE,
# scarica modello e un dataset di prompt da HuggingFace, esegue una sessione
# lunga di query registrando su JSONL gli esperti attivati per token (layer per
# layer, espansione inclusa), poi produce le statistiche di predizione con
# moe_predict.py (frequenze per layer, copertura top-M, Markov L->L+1,
# persistenza) e salva predictions.json.
#
# Uso (sulla VM):
#   bash moe-predict-run.sh
# Variabili d'ambiente:
#   MOE_MODEL_FILE     file GGUF da usare (default: UD-Q2_K_XL scaricato da unsloth)
#   MOE_MODEL_REPO     repo HF del GGUF (default unsloth/Qwen3.6-35B-A3B-GGUF)
#   MOE_DATASET        dataset HF di prompt (default: HuggingFaceH4/no_robots)
#   MOE_N_PROMPTS      numero di prompt della sessione (default: 200)
#   MOE_MAX_TOKENS     max_tokens per risposta (default: 512)
#   MOE_TOP            M della copertura top-M (default: 20)
#   MOE_PORT           porta del server (default: 9080)
#   MOE_EXTRA_FLAGS    flag extra per agrillamoe (es. "--reasoning off")
#
# Output: run.jsonl (log esperti), moe-report.txt, predictions.json
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_REPO="${MOE_MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_FILE="${MOE_MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf}"
DATASET="${MOE_DATASET:-HuggingFaceH4/no_robots}"
N_PROMPTS="${MOE_N_PROMPTS:-200}"
MAX_TOKENS="${MOE_MAX_TOKENS:-512}"
TOPM="${MOE_TOP:-20}"
PORT="${MOE_PORT:-9080}"
EXTRA="${MOE_EXTRA_FLAGS:-}"
RUN_JSONL="$HERE/run.jsonl"

echo "== [1/6] dipendenze =="
command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential cmake curl; }
pip3 install -q -U huggingface_hub datasets 2>&1 | tail -1 || true

echo "== [2/6] build AgrillaMoE (nativo per questa GPU) =="
cd "$HERE"
[ -d llama.cpp/tools/server ] || ./bootstrap-llama.sh
if [ ! -x dist/linux/agrillamoe ] || [ "${MOE_FORCE_BUILD:-0}" = "1" ]; then
    AGRILLA_JOBS="$(nproc)" ./build-linux.sh
fi

echo "== [3/6] modello e dataset =="
mkdir -p models
if [ ! -f "models/$MODEL_FILE" ]; then
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir models
fi
# prompt dal dataset HF (streaming, niente download completo); fallback interno
python3 - "$DATASET" "$N_PROMPTS" > prompts.jsonl <<'PY'
import json, sys
ds_name, n = sys.argv[1], int(sys.argv[2])
try:
    from datasets import load_dataset
    prompts = []
    ds = load_dataset(ds_name, split="train", streaming=True)
    for row in ds:
        t = row.get("prompt") or row.get("question") or row.get("instruction") or row.get("text") or ""
        if isinstance(t, list):  # chat format
            t = " ".join(m.get("content", "") for m in t if isinstance(m, dict))
        t = (t or "").strip()
        if len(t) >= 20:
            prompts.append(t[:2000])
        if len(prompts) >= n:
            break
    for p in prompts:
        print(json.dumps({"prompt": p}, ensure_ascii=False))
    print(f"dataset ok: {len(prompts)} prompt da {ds_name}", file=sys.stderr)
except Exception as e:
    print(f"dataset non disponibile ({e}): uso prompt interni", file=sys.stderr)
    base = [
        "Spiega in dettaglio il funzionamento di un transformer.",
        "Scrivi una funzione Python che calcola la mediana di una lista.",
        "Riassumi le cause della prima guerra mondiale.",
        "Descrivi il ciclo dell'acqua con precisione scientifica.",
        "Progetta un sistema di irrigazione per un orto di 100 mq.",
        "Confronta le architetture RISC e CISC con esempi.",
        "Scrivi una poesia sul mare in endecasillabi.",
        "Analizza pro e contro del lavoro remoto nel software.",
        "Come si costruisce un forno a legna in giardino?",
        "Deriva la formula della dilatazione temporale relativistica.",
    ]
    import itertools
    for i in range(n):
        print(json.dumps({"prompt": base[i % len(base)] + f" (variante {i//len(base)+1})"}, ensure_ascii=False))
PY

echo "== [4/6] avvio server con log esperti (LLAMA_MOE_EXPERT_LOG) =="
rm -f "$RUN_JSONL"
LLAMA_MOE_EXPERT_LOG="$RUN_JSONL" \
    nohup "$HERE/dist/linux/agrillamoe" \
        -m "models/$MODEL_FILE" \
        --host 127.0.0.1 --port "$PORT" --no-browser \
        -np 1 -c 16384 --temp 0.7 $EXTRA \
        > server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 200); do
    sleep 3
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "server morto"; tail -20 server.log; exit 1; }
done
echo "server pronto (pid $SRV)"

echo "== [5/6] sessione: $N_PROMPTS query x max $MAX_TOKENS token =="
python3 - "$PORT" "$MAX_TOKENS" <<'PY'
import json, sys, time, urllib.request
port, max_tokens = sys.argv[1], int(sys.argv[2])
n_done = n_err = 0
t0 = time.time()
with open("prompts.jsonl") as f:
    for i, line in enumerate(f):
        try:
            prompt = json.loads(line)["prompt"]
            body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.7,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                json.loads(r.read())
            n_done += 1
        except Exception as e:
            n_err += 1
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  {i+1} query ({n_done} ok, {n_err} err) in {dt:.0f}s", flush=True)
print(f"sessione completa: {n_done} ok, {n_err} errori")
PY

kill $SRV 2>/dev/null || true
trap - EXIT
sleep 2

echo "== [6/6] analisi e predictions =="
ls -la "$RUN_JSONL"
python3 "$HERE/moe_predict.py" "$RUN_JSONL" --top "$TOPM" --out "$HERE/predictions.json" | tee moe-report.txt
echo
echo "completato: $RUN_JSONL, moe-report.txt, $HERE/predictions.json"
