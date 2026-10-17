#!/bin/bash
# Program "llama" di supervisord: avvia AgrillaMoE. I marker log che il
# PyWorker di vast.ai attende ("model loaded" / "failed to load model") sono
# gia' le righe standard di llama-server.
set -u

ARGS=(
    -m "${MODEL_PATH:?MODEL_PATH non impostato}"
    --host 0.0.0.0
    --port "${MODEL_SERVER_PORT:-18000}"
    --no-browser
    --jinja
    -c "${LLAMA_CTX:-32768}"
    -ngl "${LLAMA_NGL:-99}"
)

# profilazione MoE automatica del wrapper: attiva di default; --no-moe-expansion
# via LLAMA_ARGS per il baseline stock
if [ "${AGRILLA_NO_MOE_EXPANSION:-0}" = "1" ]; then
    ARGS+=(--no-moe-expansion)
fi

# extra utente: es. LLAMA_ARGS="--reasoning-budget 8192 -ctk q8_0 -ctv q8_0"
# shellcheck disable=SC2086
exec /usr/local/bin/agrillamoe "${ARGS[@]}" ${LLAMA_ARGS:-}
