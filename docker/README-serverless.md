# AgrillaMoE su vast.ai Serverless (Docker)

Immagine che incapsula AgrillaMoE per gli endpoint serverless di vast.ai,
seguendo il contratto del **PyWorker** ufficiale con `BACKEND=llama`.

## Come funziona (architettura)

```
client (SDK vastai) ──> engine vast.ai ──> PyWorker (porta 3000, TLS)
                                             │  proxy OpenAI-compatibile
                                             ▼
                                  agrillamoe (porta 18000)   ← supervisord "llama"
                                  log → /var/log/portal/llama.log
```

- Il **PyWorker** di vast.ai (clonato e avviato da `start_server.sh`) taila
  `/var/log/portal/llama.log`: readiness = riga **"model loaded"**, errori =
  "failed to load model" / "INFO exited: llama" — sono le righe standard di
  llama-server/AgrillaMoE, quindi nessun adattamento necessario.
- Il backend deve stare su **porta 18000** (`MODEL_SERVER_PORT`, default del
  PyWorker) e parla OpenAI-compatibile (`/v1/chat/completions`, `/v1/completions`).
- Il wrapper AgrillaMoE **auto-inietta il profilo MoE-expansion** dalla firma
  GGUF (Qwen3.6-35B-A3B → e20 livelli 25-39; LFM2.5-24B-A2B → e8 livelli 25-39):
  sul serverless non serve configurare nulla per la profilazione.

## Variabili d'ambiente

| variabile | significato | default |
|---|---|---|
| `LLAMA_HF_REPO` + `LLAMA_HF_FILE` | modello da scaricare da HF se assente | — |
| `MODEL_URL` | alternativa: URL diretto del GGUF | — |
| `MODEL_PATH` | alternativa: percorso di un modello su volume | auto (il più grande in `MODELS_DIR`) |
| `LLAMA_CTX` | contesto | `32768` |
| `LLAMA_NGL` | layer su GPU | `99` |
| `LLAMA_ARGS` | extra (es. `--reasoning-budget 8192 -ctk q8_0 -ctv q8_0`) | — |
| `AGRILLA_NO_MOE_EXPANSION` | `1` = baseline stock (senza expansion) | `0` |
| `MODEL_SERVER_PORT` | porta backend verso il PyWorker | `18000` |
| `HF_TOKEN` | token HF per repo gated | — |

## Setup dell'endpoint (dopo la pubblicazione su GHCR)

1. **Template** nuovo su vast.ai: immagine
   `ghcr.io/vagrillo/agrillamoe-serverless:latest`, porta server `18000` (http).
2. **Env** del template: `LLAMA_HF_REPO=LiquidAI/LFM2.5-24B-A2B-GGUF`,
   `LLAMA_HF_FILE=LFM2.5-24B-A2B-QAD-Q4_0.gguf` (o il Qwen3.6 che serve),
   eventuale `HF_TOKEN`, `LLAMA_ARGS=--reasoning-budget 8192 -ctk q8_0 -ctv q8_0`.
3. **Workergroup/endpoint**: workload type "LLM Inference"; monta un **volume**
   su `/workspace/models` per non ri-scaricare il modello a ogni cold start
   (il download HF di 13-25 GB domina il cold start altrimenti).
4. Chiamata dal client (OpenAI-compatibile):

```python
from vastai import Serverless
client = Serverless()          # VAST_API_KEY nell'env
ep = await client.get_endpoint(name="agrillamoe-lfm25")
r = await ep.request("/v1/chat/completions", {
    "model": "LFM2.5-24B-A2B-QAD-Q4_0",
    "messages": [{"role": "user", "content": "2+2?"}],
    "max_tokens": 4096,
}, cost=4096)
```

## Note operative

- **Cold start**: download modello (se niente volume) + load + warmup; con
  volume montato resta solo il load (~1-2 min per 13-24 GB). Aumenta
  `cold_workers`/`min_load` dell'endpoint per tenerne uno caldo.
- **GPU**: l'immagine include CUDA arch 70/80/86/90 (V100, A100, 3090/4090,
  H100); il binario è built con baseline CPU portabile (`AGRILLA_NATIVE=0`).
- **Build locale**: `docker build -t agrillamoe-serverless -f docker/Dockerfile .`
- **Push manuale**: `docker tag ... ghcr.io/vagrillo/agrillamoe-serverless:latest && docker push ...`
  (il workflow `docker-serverless` lo fa in automatico su ogni push a main).
