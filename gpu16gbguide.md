# AgrillaMoE — Guida GPU 16 GB / 16 GB GPU Guide..12GB? maybe!

> **Qwen3.6-35B-A3B quantized 2-bit with MoE-expansion, a step-by-step guide for non-experts.**
> In this document: [English version](#english) · [versione italiana](#italiano)

---


<a name="english"></a>
# 🇬🇧 English version

AgrillaMoE runs **Qwen3.6-35B-A3B** (a very capable MoE model) on your own PC
using the **2-bit quant (UD-Q2_K_XL, ~11.4 GB)** and enables **MoE-expansion**:
a technique that activates more of the model's "experts" per token, which in
the GPQA-Diamond benchmark lifts accuracy to 84.3% (vs 81.8% stock). You need
an **NVIDIA GPU with at least 16 GB VRAM** (e.g. V100 16GB, RTX 4080/5080),
an up-to-date driver and ~15 GB of disk. Smaller GPUs still work, but part of
the model goes to system RAM (slower).

## 1. Install prerequisites (one time)

1. **NVIDIA driver**: get the latest from <https://www.nvidia.com/drivers>.
2. **Verify**: open a terminal (Windows: `Win+R` → `powershell`; Linux: your
   Terminal app) and run:

   ```
   nvidia-smi
   ```

   You should see your GPU name and "CUDA Version: 12.x" or newer.

## 2. Download AgrillaMoE

Go to <https://github.com/vagrillo/AgrillaMoE/releases> and get the asset for
your system:

- **Windows**: `agrillamoe-…-windows-x64-multiarch.zip`
- **Linux**: `agrillamoe-…-linux-x86_64-universal.tar.xz`

### Windows
1. Right-click the `.zip` → **Extract All** → to a simple folder such as
   `C:\AgrillaMoE`.
2. **Windows-only note**: the program needs the `cublas64_12.dll` and
   `cublasLt64_12.dll` libraries. If a "DLL not found" error appears at
   startup, install the **CUDA Toolkit 12.x** from
   <https://developer.nvidia.com/cuda-downloads> (Windows, "Express"
   install) and retry.

### Linux
In a terminal:

```bash
mkdir -p ~/AgrillaMoE && cd ~/AgrillaMoE
tar -xf ~/Downloads/agrillamoe-*-linux-x86_64-universal.tar.xz   # adapt path
chmod +x agrillamoe
```

## 3. Start AgrillaMoE (the easy part)

Open a terminal **in the AgrillaMoE folder**:

- **Windows** (PowerShell): `.\agrillamoe.exe`
- **Linux**: `./agrillamoe`

What happens:

1. A **menu** lists every quant (compressed version) of the model. For a
   16 GB GPU the best one is pre-selected: pick **UD-Q2_K_XL (~11.4 GB)** —
   just press Enter on the line marked `<== consigliato per la tua VRAM`
   (recommended for your VRAM).
2. If the file is not on disk yet, you'll be asked `Scarico … ? [s/N]`
   ("Download?"): type `s` and Enter. The ~11.4 GB download starts if the
   `hf` command is installed (see 3bis).
3. When loading finishes, the server starts and **the browser opens
   automatically** at <http://127.0.0.1:8071> with a ready-to-use chat UI.

### 3bis. If the automatic download doesn't start

It uses Python's `hf` tool. Install it with:

- Windows: `pip install -U huggingface_hub`
- Linux: `pip3 install -U huggingface_hub` (or `pipx install huggingface_hub`)

Or download the model **manually** (no Python needed) from:
<https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf>
and place it in:

- Windows: `C:\Users\YOURNAME\models\`
- Linux: `~/models/`

Then start AgrillaMoE again: the file will show up in the menu.

## 4. Use the model

- **Chat**: type in the browser page at <http://127.0.0.1:8071>.
- **API** (Open WebUI, LibreChat, scripts): endpoint
  `http://127.0.0.1:8071/v1`, OpenAI-compatible.

Example with `curl` (same on Windows and Linux):

```bash
curl http://127.0.0.1:8071/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi! Who are you?"}]}'
```

## 5. Defaults set for you (no action needed)

| What | Value | Why |
|---|---|---|
| MoE-expansion | 20 experts, threshold 0.80, layers 25–39 | the recipe that scores 84.3% on GPQA-Diamond in benchmarks |
| Context | 142,768 tokens (~140k) across 4 slots | same as the benchmarks; auto-shrinks if VRAM is tight |
| Endpoint | `127.0.0.1:8071` + auto browser | everything stays on your PC |

## 6. Useful options (optional)

Append these to the start command:

- `--reasoning off` → instant answers without the thinking phase (by default
  the model reasons before answering, and at 2-bit it reasons a lot)
- `--reasoning-budget 4096` → keep reasoning but cap its length
- `-c 32768 -np 1` → smaller context and a single slot (less VRAM)
- `--no-browser` → don't open the browser
- `-m path/to/model.gguf` → skip the menu and load that file

Stop the server with `Ctrl+C` in the terminal.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "DLL not found" (Windows) | install CUDA Toolkit 12.x (step 2) |
| Browser doesn't open | go to <http://127.0.0.1:8071> manually |
| Very slow | GPU below 16 GB: expected — part of the model lives in RAM; try `--reasoning off` and `-c 8192` |
| "no GPU / CUDA error" | update the NVIDIA driver (must support CUDA 12+) |
| Interrupted download | start AgrillaMoE again and answer `s`: the download resumes |

## 8. Using coding agents with AgrillaMoE (Claude Code and more)

AgrillaMoE exposes both **OpenAI-compatible** APIs (`/v1/chat/completions`)
and the **Anthropic API** (`/v1/messages`, the one Claude Code speaks). Keep
the AgrillaMoE terminal open and use the agent in a second one.

### Claude Code

1. Install Claude Code (requires Node.js 18+): `npm install -g @anthropic-ai/claude-code`
2. Open a **second** terminal (leave AgrillaMoE running) and type:

   **Linux / macOS:**
   ```bash
   export ANTHROPIC_BASE_URL=http://127.0.0.1:8071
   export ANTHROPIC_AUTH_TOKEN=agrilla
   unset ANTHROPIC_API_KEY
   claude
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8071"
   $env:ANTHROPIC_AUTH_TOKEN = "agrilla"
   Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
   claude
   ```

3. Answer Claude Code's first-run questions, then give it a task
   (`claude "explain this project"` or inside the chat). The token can be any
   string (the server does not require keys).

Tip: if answers feel slow because of the model's long thinking, restart
AgrillaMoE with `--reasoning-budget 4096` (caps the thinking) or
`--reasoning off` (direct answers).

### Other open-source coding agents (OpenAI-compatible endpoint)

Base URL: `http://127.0.0.1:8071/v1` — API key: any value.

| Agent | How to connect |
|---|---|
| **Aider** | `export OPENAI_API_KEY=x` then `aider --model openai/qwen3.6-35b --openai-api-base http://127.0.0.1:8071/v1` |
| **OpenCode** | `opencode` → provider "OpenAI compatible" → Base URL `http://127.0.0.1:8071/v1`, model `qwen3.6-35b` |
| **Cline / Roo Code** (VS Code) | Settings → API Provider: **OpenAI Compatible** → Base URL `http://127.0.0.1:8071/v1` → model `qwen3.6-35b` |
| **Continue** (VS Code/JetBrains) | config: provider `openai`, `apiBase: http://127.0.0.1:8071/v1`, model `qwen3.6-35b` |
| **Zed** | settings.json → OpenAI-compatible provider with `api_url: http://127.0.0.1:8071/v1` |
| **Goose** | `GOOSE_PROVIDER=openai`, `OPENAI_HOST=http://127.0.0.1:8071/v1/`, `GOOSE_MODEL=qwen3.6-35b` |

See the exact loaded model name with:
`curl http://127.0.0.1:8071/v1/models` (any name works anyway).

---

*Guida relativa ad AgrillaMoE v1.0.1 — [repository](https://github.com/vagrillo/AgrillaMoE).*

---

<a name="italiano"></a>
# 🇮🇹 Versione italiana (Italian version)

AgrillaMoE fa girare sul tuo PC il modello **Qwen3.6-35B-A3B** (un modello
"MoE" molto potente) nella versione **quantizzata a 2 bit (UD-Q2_K_XL,
~11,4 GB)**, attivando la **MoE-expansion**: una tecnica che usa più "esperti"
del modello a ogni token e che nei benchmark GPQA-Diamond porta l'accuratezza
dell'84,3% (contro l'81,8% del modello normale). Serve una **GPU NVIDIA con
almeno 16 GB di VRAM** (es. V100 16GB, RTX 4080, RTX 5080, RX… no, solo
NVIDIA 😄), driver aggiornato e ~15 GB di spazio disco. Con GPU più piccole
funziona lo stesso, ma una parte del modello va in RAM (più lento).

## 1. Installa i prerequisiti (una volta sola)

1. **Driver NVIDIA aggiornato**: apri il sito
   <https://www.nvidia.com/drivers> e installa l'ultimo driver per la tua GPU.
2. **Verifica**: apri un terminale (Windows: premi `Win+R`, scrivi `powershell`,
   Invio; Linux: apri la finestra "Terminale") e scrivi:

   ```
   nvidia-smi
   ```

   Se vedi una tabella con il nome della tua GPU e "CUDA Version: 12.x" o
   superiore, sei a posto.

## 2. Scarica AgrillaMoE

Vai su <https://github.com/vagrillo/AgrillaMoE/releases> e scarica l'ultimo file
"asset" per il tuo sistema:

- **Windows**: `agrillamoe-…-windows-x64-multiarch.zip`
- **Linux**: `agrillamoe-…-linux-x86_64-universal.tar.xz`

### Windows
1. Clicca con il tasto destro sul file `.zip` → **Estrai tutto** → in una
   cartella semplice, ad esempio `C:\AgrillaMoE`.
2. **Importante (solo Windows)**: il programma ha bisogno di due librerie
   `cublas64_12.dll` e `cublasLt64_12.dll`. Se sono già nel tuo sistema
   (installando il driver/CUDA spesso ci sono) tutto funziona; se all'avvio
   vedi un errore "DLL non trovata", installa il **CUDA Toolkit 12.x** da
   <https://developer.nvidia.com/cuda-downloads> (scegli Windows, exe
   "Express") e riprova.

### Linux
Apri il terminale nella cartella degli Scaricamenti e scrivi (usa il TAB per
completare il nome del file):

```bash
mkdir -p ~/AgrillaMoE && cd ~/AgrillaMoE
tar -xf ~/Scaricamenti/agrillamoe-*-linux-x86_64-universal.tar.xz   # adatta il percorso
chmod +x agrillamoe
```

## 3. Avvia AgrillaMoE (la parte facile)

Apri il terminale **nella cartella di AgrillaMoE**:

- **Windows** (PowerShell): `.\agrillamoe.exe`
- **Linux**: `./agrillamoe`

Cosa succede:

1. Compare un **menu** con tutti i "quant" (le versioni compresse) del modello.
   Per una GPU da 16 GB il programma ti propone già il migliore: scegli
   **UD-Q2_K_XL (~11,4 GB)** e premi Invio, oppure digita il numero della
   riga con scritto `<== consigliato per la tua VRAM`.
2. Se il file non è ancora sul tuo disco, il programma chiede
   `Scarico … ? [s/N]`: scrivi `s` e Invio. Il download (~11,4 GB) parte se
   hai installato il comando `hf` (vedi sotto); altrimenti vedi il punto 3bis.
3. Alla fine il server parte e **il browser si apre da solo** su
   <http://127.0.0.1:8071>: hai una chat pronta (interfaccia web).

### 3bis. Se il download automatico non parte

Il download usa lo strumento `hf` di Python. Se non c'è, installalo con:

- Windows: `pip install -U huggingface_hub`
- Linux: `pip3 install -U huggingface_hub` (oppure `pipx install huggingface_hub`)

Oppure scarica il modello **a mano** (senza Python) dal browser:
<https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf>
e mettilo in:

- Windows: `C:\Users\TUONOME\models\`
- Linux: `~/models/`

Poi riavvia AgrillaMoE: il file verrà trovato e ti basterà sceglierlo dal menu.

## 4. Usare il modello

- **Chat**: nel browser su <http://127.0.0.1:8071> scrivi e Invio.
- **API** (per programmi tipo Open WebUI, LibreChat, script): l'endpoint è
  `http://127.0.0.1:8071/v1`, compatibile con le API di OpenAI.

Esempio con `curl` (funziona uguale su Windows e Linux):

```bash
curl http://127.0.0.1:8071/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Ciao! Chi sei?"}]}'
```

## 5. Le impostazioni già pronte per te (non toccare nulla)

AgrillaMoE configura da solo, a ogni avvio:

| Cosa | Valore | Perché |
|---|---|---|
| MoE-expansion | 20 esperti, soglia 0,80, livelli 25–39 | la ricetta che nei benchmark dà 84,3% su GPQA-Diamond |
| Contesto | 142.768 token (~140k) su 4 slot | come i benchmark; se la VRAM non basta si riduce da solo |
| Endpoint | `127.0.0.1:8071` + browser automatico | tutto resta sul tuo PC |

## 6. Opzioni utili (facoltative)

Da aggiungere dopo il comando di avvio:

- `--reasoning off` → risposte immediate, senza "pensiero" (il modello di
  default ragiona prima di rispondere, e i 2-bit ragionano a lungo: questa
  opzione lo disattiva)
- `--reasoning-budget 4096` → lascia il ragionamento ma limita la lunghezza
- `-c 32768 -np 1` → contesto più piccolo e un solo "slot" (meno VRAM usata)
- `--no-browser` → non aprire il browser
- `-m percorso/del/modello.gguf` → salta il menu e usa quel file

Per fermare il server: premi `Ctrl+C` nella finestra del terminale.

## 7. Problemi comuni

| Sintomo | Cosa fare |
|---|---|
| "DLL non trovata" (Windows) | installa il CUDA Toolkit 12.x (punto 2) |
| Non si apre il browser | vai a mano su <http://127.0.0.1:8071> |
| Tutto molto lento | GPU con meno di 16 GB: normale, parte del modello va in RAM; prova `--reasoning off` e `-c 8192` |
| "nessuna GPU / CUDA error" | aggiorna il driver NVIDIA (deve supportare CUDA 12+) |
| Scaricamento interrotto | rilancia AgrillaMoE e ridai `s`: il download riprende da dove era arrivato |

## 8. Usare coding agent con AgrillaMoE (Claude Code e altri)

AgrillaMoE espone sia le API **OpenAI-compatibili** (`/v1/chat/completions`)
sia l'**API Anthropic** (`/v1/messages`, usata da Claude Code). Il server resta
in esecuzione mentre usi l'agent in un'altra finestra del terminale.

### Claude Code

1. Installa Claude Code (serve Node.js 18+): `npm install -g @anthropic-ai/claude-code`
2. Apri un **secondo** terminale (AgrillaMoE deve restare acceso) e digita:

   **Linux / macOS:**
   ```bash
   export ANTHROPIC_BASE_URL=http://127.0.0.1:8071
   export ANTHROPIC_AUTH_TOKEN=agrilla
   unset ANTHROPIC_API_KEY
   claude
   ```

   **Windows (PowerShell):**
   ```powershell
   $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8071"
   $env:ANTHROPIC_AUTH_TOKEN = "agrilla"
   Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
   claude
   ```

3. Alla prima esecuzione rispondi alle domande di Claude Code, poi scrivi il
   tuo task (`claude "spiega questo progetto"` oppure dentro la chat).
   Il token è un valore qualsiasi (il server non richiede chiavi).

Consiglio: se le risposte sono lente per il lungo ragionamento del modello,
riavvia AgrillaMoE con `--reasoning-budget 4096` (limita il pensiero) oppure
`--reasoning off` (risposte dirette).

### Altri coding agent open source (endpoint OpenAI-compatibile)

Base URL da usare: `http://127.0.0.1:8071/v1` — API key: qualsiasi valore.

| Agent | Come si collega |
|---|---|
| **Aider** | `export OPENAI_API_KEY=x` poi `aider --model openai/qwen3.6-35b --openai-api-base http://127.0.0.1:8071/v1` |
| **OpenCode** | `opencode` → provider "OpenAI compatible" → Base URL `http://127.0.0.1:8071/v1`, model `qwen3.6-35b` |
| **Cline / Roo Code** (VS Code) | Impostazioni → API Provider: **OpenAI Compatible** → Base URL `http://127.0.0.1:8071/v1` → model `qwen3.6-35b` |
| **Continue** (VS Code/JetBrains) | config: provider `openai`, `apiBase: http://127.0.0.1:8071/v1`, model `qwen3.6-35b` |
| **Zed** | settings.json → provider OpenAI-compatible con `api_url: http://127.0.0.1:8071/v1` |
| **Goose** | `GOOSE_PROVIDER=openai`, `OPENAI_HOST=http://127.0.0.1:8071/v1/`, `GOOSE_MODEL=qwen3.6-35b` |

Il nome esatto del modello caricato lo vedi con:
`curl http://127.0.0.1:8071/v1/models` (qualsiasi nome funziona comunque).

---

