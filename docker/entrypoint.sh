#!/bin/bash
# Entrypoint AgrillaMoE per vast.ai Serverless.
#
# 1. scarica il modello se non presente (LLAMA_HF_REPO+LLAMA_HF_FILE o MODEL_URL)
# 2. avvia agrillamoe sotto supervisord (log → /var/log/portal/llama.log:
#    il PyWorker di vast.ai riconosce "model loaded" come readiness)
# 3. lancia lo start_server.sh ufficiale di vast.ai/PyWorker (BACKEND=llama:
#    dedotto automaticamente da LLAMA_MODEL impostato qui sotto)
#
# Il wrapper AgrillaMoE auto-inietta il profilo MoE-expansion in base alla firma
# GGUF (Qwen3.6 → e20, LFM2.5 → e8, layer 25-39): per serverless non serve
# passare flag extra. LLAMA_ARGS permette override/extra (es. --reasoning-budget 8192).
set -e

MODELS_DIR="${MODELS_DIR:-/workspace/models}"
mkdir -p "$MODELS_DIR" /var/log/portal

# ---- risoluzione modello -------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"

if [ -z "$MODEL_PATH" ] && [ -n "$MODEL_URL" ]; then
    MODEL_PATH="$MODELS_DIR/$(basename "${MODEL_URL%%\?*}")"
fi
if [ -z "$MODEL_PATH" ] && [ -n "${LLAMA_HF_REPO:-}" ] && [ -n "${LLAMA_HF_FILE:-}" ]; then
    MODEL_PATH="$MODELS_DIR/$(basename "$LLAMA_HF_FILE")"
    if [ ! -s "$MODEL_PATH" ]; then
        echo "[serverless] download $LLAMA_HF_REPO/$LLAMA_HF_FILE"
        hdr=()
        [ -n "${HF_TOKEN:-}" ] && hdr=(-H "Authorization: Bearer $HF_TOKEN")
        curl -sL --fail "${hdr[@]}" -o "$MODEL_PATH.part" \
            "https://huggingface.co/$LLAMA_HF_REPO/resolve/main/$LLAMA_HF_FILE"
        mv "$MODEL_PATH.part" "$MODEL_PATH"
    fi
fi
if [ -z "$MODEL_PATH" ]; then
    # nessun modello specificato: prende il piu' grande gia' presente nel volume
    MODEL_PATH="$(ls -S "$MODELS_DIR"/*.gguf 2>/dev/null | head -1 || true)"
fi
if [ -z "$MODEL_PATH" ] || [ ! -s "$MODEL_PATH" ]; then
    echo "[serverless] ERRORE: nessun modello (imposta LLAMA_HF_REPO+LLAMA_HF_FILE, MODEL_URL o monta un volume in $MODELS_DIR)" >&2
    exit 1
fi
export MODEL_PATH
echo "[serverless] modello: $MODEL_PATH"

# ---- nome modello per il PyWorker (benchmark/telemetria) ------------------
if [ -z "${LLAMA_MODEL:-}" ]; then
    export LLAMA_MODEL="$(basename "$MODEL_PATH" .gguf)"
fi

# ---- PyWorker: prepara start_server.sh (in image, fallback a runtime) -----
if [ ! -f /opt/start_server.sh ]; then
    curl -sSL https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh \
        -o /opt/start_server.sh
fi
chmod +x /opt/start_server.sh

# ---- supervisord: avvia agrillamoe come program "llama" -------------------
supervisord -c /etc/supervisor/conf.d/llama.conf &

# ---- PyWorker (BACKEND=llama, porta 18000, proxy OpenAI-compatibile) ------
export BACKEND="${BACKEND:-llama}"
exec bash /opt/start_server.sh
