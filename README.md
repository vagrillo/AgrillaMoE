# AgrillaMoE

**AgrillaMoE** is a dedicated `llama-server` build for **Qwen3.6-35B-A3B (MoE)**
inference with the quantized GGUFs published by **Unsloth**, built on the
[`vagrillo/llama.cpp`](https://github.com/vagrillo/llama.cpp) fork (branch
`moe-expansion`) which implements runtime expansion of the routed experts.

> 📘 **First time with a 16 GB GPU?** Read **[gpu16gbguide.md](gpu16gbguide.md)** —
> a step-by-step guide (English + italiano) to run Qwen3.6-35B-A3B 2-bit with
> MoE-expansion on Windows and Linux.

```
    _                    _      _  __  __  _   _ ___
   / \   __ _  ___ _ __ | |    / \|  \/  |/ / | |_ _|
  / _ \ / _` |/ _ \ '_ \| |   / _ \ |\/| | || | || |
 / ___ \ (_| |  __/ | | | |  / ___ \ |  | | || | || |
/_/   \_\__, |\___|_| |_|_| /_/   \_\_|  |_| \_/|___|
        |___/  dedicated Qwen3.6-35B-A3B inference server
```

## What happens at startup

1. **Model selection**: if a Qwen3.6-35B-A3B GGUF is already downloaded (in
   `$AGRILLA_MODELS_DIR`, `~/models`, `./models`, `C:\models` / `/mnt/c/models`)
   it asks which one to use; if nothing exists yet, it suggests the largest
   Unsloth quant that fits the **detected GPU VRAM** (via `nvidia-smi`, ~92% of
   VRAM as budget) and downloads it with the `hf` CLI after confirmation.
   `Enter` always picks the recommendation; `d` = download another quant,
   `x` = cancel.
2. **Default MoE-expansion profile** — exactly the one used in the benchmarks
   with **Qwen3.6-35B-A3B Q8_0** (GPQA-Diamond: **84.34%** with expansion vs
   81.82% stock top-8):

   | parameter | value |
   |---|---|
   | `--moe-experts` | 20 (model default 8) |
   | `--moe-expert-threshold` | 0.80 (adaptive: 5..20 experts/token) |
   | `--moe-expert-layer-start` | 25 |
   | `--moe-expert-layer-end` | 39 (of 40 layers) |
   | decay / renorm | 0.50 / auto (fork defaults) |

   The profile is injected **only if the user passes no `--moe-*` / `--q35-*`
   flag**; `--no-moe-expansion` (or `AGRILLA_NO_MOE_EXPANSION=1`) disables it
   entirely.
3. **Default endpoint `127.0.0.1:8071`** and **automatic browser open**
   (Windows via `ShellExecute`, Linux/WSL via `wslview`) as soon as the server
   listens. `--no-browser` or `AGRILLA_NO_BROWSER=1` disables it.
4. Other defaults injected only when missing: `--jinja` (Qwen chat template),
   `-c 142768` (~140k context) and `--parallel 4` (4 slots × 35840 tokens —
   the RUN1209/Q2 benchmark configuration). On low-VRAM GPUs the fork's
   `fit_params` automatically shrinks the context to fit model + KV cache.

Every other `llama-server` flag passes straight through: context, temperature,
concurrency and sampling are set with the standard flags, e.g. `-c 8192`,
`--temp 0.6`, `--top-p 0.95`, `-np 8`, `--threads 8` (`--help` for the full list).

### AgrillaMoE-specific flags

| flag / env | effect |
|---|---|
| `--agrilla-list-models` | list local models + VRAM recommendation, then exit |
| `--agrilla-models-dir DIR` | add a model search folder |
| `--agrilla-yes` / `AGRILLA_YES=1` | auto-confirm downloads |
| `--no-browser` / `AGRILLA_NO_BROWSER=1` | don't open the browser |
| `--no-moe-expansion` / `AGRILLA_NO_MOE_EXPANSION=1` | stock top-8 routing |
| `AGRILLA_MODELS_DIR` | folder list (`:` on Linux, `;` on Windows) |
| `AGRILLA_REASONING_BUDGET` | injects `--reasoning-budget N` (e.g. `8192`; `-1` unlimited, `0` ends thinking immediately) |
| `AGRILLA_REASONING` | injects `--reasoning on\|off\|auto` (e.g. `off` disables thinking entirely) |
| `--agrilla-streaming` / `AGRILLA_STREAMING=1` | **streaming mode** (see below) for low-VRAM GPUs |
| `--agrilla-gpu-streaming` / `AGRILLA_GPU_STREAMING=1` | all layers on GPU, weights paged RAM↔VRAM by the driver |
| `--agrilla-chunk-predict` | chunked decode with routing prediction + expert prefetch (DS4-style) |

### Streaming mode (`--agrilla-streaming`) — low-VRAM GPUs

Inspired by antirez's DS4 project (weight streaming with a memory window):
designed for GPUs like a 4GB GTX 1050 with a small quant (UD-IQ1_M ~9.4 GB)
on disk. At startup AgrillaMoE configures:

- `--cpu-moe`: **expert** weights stay on disk and arrive via mmap; at each
  token `mul_mat_id` computes **only the experts chosen by the router** (the
  statistical selection is already done by the model)
- `-ngl 99`: attention, norms and KV cache go to the GPU (shrunk by
  `fit_params` if VRAM is tight)
- compact context (8192, 1 slot) unless specified otherwise
- a **sequential prefetch thread** reads the GGUF ahead of time, pulling
  upcoming layers into the RAM cache (the GGUF layout is sequential per
  layer), so page faults don't wait on the SSD

On Windows there's a ready-made launcher, `agrillamoe-lite.cmd` (uses
`D:\models`). **Per-expert** prediction of upcoming layers (a predictive model
trained on JSONL activation logs) is described in `moe-predict.md`.

### Generation parameters (native llama-server flags, all pass through)

- **Context/concurrency**: `-c 142768` (default), `-np 4` (default), e.g. `-c 65536 -np 2`
- **Reasoning**: `--reasoning on|off|auto` — `off` **disables thinking**; `--reasoning-budget N` — `-1` unlimited (default), `0` ends thinking immediately, `N>0` caps thinking tokens; `--reasoning-budget-message "..."` message injected at end of thinking
- **Temperature/sampling**: `--temp 0.6`, `--top-p 0.95`, `--top-k`, `--min-p`, `--repeat-penalty`...
- The `[AgrillaMoE] avvio llama-server con:` summary shows the effective context, slot, reasoning state and budget.

### VRAM suggestion (unsloth/Qwen3.6-35B-A3B-GGUF catalog)

| VRAM | suggested quant | size |
|---|---|---|
| 8 GB | UD-IQ1_M | ~9.4 GB |
| 12 GB | UD-Q2_K_XL | ~11.4 GB |
| 16 GB | UD-Q3_K_XL | ~15.7 GB |
| 24 GB | UD-Q5_K_M | ~24.6 GB |
| 32 GB | UD-Q6_K_XL | ~29.7 GB |
| 40+ GB | Q8_0 (benchmark reference) | ~34.4 GB |

If no quant fits in VRAM, the smallest one (UD-IQ1_M) is proposed with partial
CPU offload.

## Build

The llama.cpp source is not included: you need the `moe-expansion` fork branch
(looked up in `../repo` or `./llama.cpp` created by `bootstrap-llama.sh`, or
forced with `-DAGRILLA_LLAMA_DIR=...`).

### Linux (statically linked, CUDA 12+)

```bash
./bootstrap-llama.sh        # first time only, if ../repo doesn't exist
./build-linux.sh             # output: dist/linux/agrillamoe
```

Variables: `AGRILLA_CUDA_ARCH` (default `native`; e.g. `61`, `70;80;86`),
`AGRILLA_NATIVE` (`1` default = build host's `-march=native`; `0` = portable
CPU baseline, for redistributable binaries), `AGRILLA_BUILD_DIR`,
`AGRILLA_JOBS`. Requirements: cmake ≥ 3.24, gcc, CUDA toolkit 12+.

**Universal portable binary** (redistributable, all NVIDIA ≥8 GB from RTX 20xx
on plus Pascal/Volta):

```bash
AGRILLA_NATIVE=0 AGRILLA_CUDA_ARCH="61;70;75;80;86;89;120" ./build-linux.sh
```

(sm_120 = RTX 50xx requires CUDA toolkit ≥ 12.8; the last listed arch is also
included as PTX, so newer GPUs work via driver JIT.)

### Windows (statically linked, CUDA 12+)

```powershell
powershell -ExecutionPolicy Bypass -File build-windows.ps1   # output: dist\windows\agrillamoe.exe
# portable multi-arch:
powershell -ExecutionPolicy Bypass -File build-windows.ps1 -CudaArch "61;75;86;89" -Native 0 -Jobs 3
```

Parameters: `-CudaArch` (default `native`), `-Native` (default 1), `-Jobs`
(default 4), `-BuildDir`. Requirements: Visual Studio 2019 BuildTools (VC++ +
bundled CMake/Ninja) and CUDA toolkit 12.x. For RTX 50xx (sm_120) you need
CUDA toolkit ≥ 12.8: `-CudaArch "61;75;86;89;120"`.

**Release binary GPU coverage**: Linux = sm 61, 70, 75, 80, 86, 89, 120 + PTX
(GTX 10xx, V100, RTX 20xx/30xx/40xx/50xx, A100); Windows = sm 61, 75, 86, 89 +
PTX 89 (GTX 10xx, RTX 20xx/30xx/40xx; RTX 50xx via recompile with CUDA ≥ 12.8).

> Note: don't run a binary compiled for one arch only on a different GPU —
> even though PTX allows startup, kernels can crash at first generation
> (runtime dispatch vs compile-time guards). Release binaries include real
> cubins for all listed architectures.

## AMD GPUs (e.g. Radeon RX 7800 XT) and others

AgrillaMoE also runs on AMD via the **Vulkan** backend (and, recompiling, via
**HIP/ROCm**). MoE-expansion is routing logic in the graph: it works the same
on every backend (CUDA, Vulkan, ROCm, CPU).

- **Vulkan (recommended, works out of the box)**: build with the Vulkan
  backend alongside CUDA — one executable for both NVIDIA and AMD; on an AMD
  machine the server uses the GPU via Vulkan (RDNA3 included, RX 7800 XT =
  gfx1101). Only a driver with Vulkan runtime is needed (always present with
  AMD/NVIDIA drivers). Windows build: `build-windows.ps1 -Vulkan 1 ...`;
  Linux: `AGRILLA_VULKAN=1 ./build-linux.sh`. At startup `--device Vulkan0`
  forces the Vulkan GPU when several are present (`--list-devices` to list).
- **HIP/ROCm (best AMD performance, dedicated build)**: requires AMD HIP
  SDK/ROCm; Windows: `build-windows.ps1 -Hip 1 -AmdTargets gfx1101`;
  Linux: `AGRILLA_HIP=1 AGRILLA_AMD_TARGETS=gfx1101 ./build-linux.sh`
  (RX 7800 XT = `gfx1101`; `gfx1100` = 7900 XTX/XT).

Test status: NVIDIA tested directly (CUDA and Vulkan); **AMD not tested on our
hardware** — the Vulkan path is identical on every GPU, but feedback from AMD
users (especially RX 7800 XT) is welcome: open a GitHub issue with GPU, driver
and the output of `agrillamoe --list-devices`.

## macOS (Apple Silicon, Metal backend)

On M1/M2/M3/M4 Macs AgrillaMoE uses llama.cpp's **Metal** backend:
MoE-expansion works identically (routing in the graph, `mul_mat_id` supported
by Metal) and **unified memory** simplifies everything — the "budget" is total
RAM, streaming mode is rarely needed:

| Mac RAM | suggested quant |
|---|---|
| 16 GB | UD-IQ2_M / UD-Q2_K_XL |
| 24 GB | UD-Q4_K_XL |
| 32 GB | UD-Q6_K_XL |
| 48+ GB | Q8_0 whole |

Build on the Mac (requires `xcode-select --install` and `brew install cmake`):

```bash
./bootstrap-llama.sh && ./build-macos.sh    # output: dist/macos/agrillamoe
```

**Prebuilt binaries**: the `build-binaries` GitHub Action compiles on Apple
Silicon runners for every `v*` tag and attaches
`agrillamoe-…-macos-arm64-metal.tar.gz` to the release; it can also be run
manually (Actions tab → Run workflow).

> Note on **MLX**: AgrillaMoE is based on llama.cpp/C++; the Metal backend is
> the native path on Mac with performance comparable to MLX on GGUF. A port of
> MoE-expansion to mlx-lm (Python) would be a separate project — see
> `moe-predict.md` for the reusable analysis part.

## Usage

```bash
./agrillamoe                       # interactive menu, AgrillaMoE defaults
./agrillamoe -m ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
./agrillamoe --host 0.0.0.0 --port 9000 --no-browser
./agrillamoe -c 65536 -np 2        # custom context/concurrency
./agrillamoe --reasoning off       # disable thinking
./agrillamoe --reasoning-budget 8192   # cap thinking at 8192 tokens
./agrillamoe --temp 0.6 --top-p 0.95
./agrillamoe --moe-experts 12 --moe-expert-threshold 0.7   # custom profile
```

API: OpenAI-compatible at `http://127.0.0.1:8071/v1` (+ web UI at the root),
identical to `llama-server`; it also exposes the **Anthropic
`/v1/messages`** endpoint, so **Claude Code** and OpenAI-compatible agents
(Aider, Cline, OpenCode, Continue, Zed, Goose) connect directly — step-by-step
instructions in [gpu16gbguide.md](gpu16gbguide.md), section 8.

## Structure

```
AgrillaMoE/
├── CMakeLists.txt        # standalone project adding the fork as a subdirectory
├── src/main.cpp          # banner, model/VRAM selection, hf download, defaults, browser
├── build-linux.sh        # static CUDA Linux build
├── build-windows.ps1     # static CUDA Windows build (VS2019 BuildTools + CUDA 12)
├── build-macos.sh        # static Metal macOS build
├── bootstrap-llama.sh    # clones the moe-expansion fork when needed
└── dist/                 # built binaries (not versioned)
```

## Benchmark results: HumanEval

First public HumanEval measurement for Qwen3.6-35B-A3B — expansion vs stock
routing A/B on an RTX 2080 Ti with UD-IQ4_XS 4-bit (thinking budget 4096):

| Config | pass@1 | Decode |
|---|---|---|
| Stock top-8 | 89.63% (147/164) | 69.3 tok/s |
| MoE-expansion 20 | **90.85%** (149/164) | 56.0 tok/s |

Full details (methodology, paired analysis, caveats) in
**[humaneval-eval.md](humaneval-eval.md)**.

## Background

- MoE expansion: runtime routing with more routed experts than the native
  top-K (see `docs/moe-expansion.md` in the fork).
- Reference benchmark (RUN1209, vast.ai V100): Qwen3.6-35B-A3B **Q8_0** with
  `N=20 T=0.8 L25–39` → GPQA-Diamond 84.34% vs 81.82% stock (+2.5 points).
  This is the profile AgrillaMoE injects by default.

## License

MIT (like llama.cpp). This project includes/modifies code from
[llama.cpp](https://github.com/ggml-org/llama.cpp) and the
`vagrillo/llama.cpp` fork (moe-expansion branch).

---

# Versione italiana (Italian version)

*Contenuto identico alla versione inglese.*

**AgrillaMoE** è una versione dedicata di `llama-server` specializzata
nell'inferenza di **Qwen3.6-35B-A3B (MoE)** con i GGUF quantizzati pubblicati da
**Unsloth**, costruita sul fork
[`vagrillo/llama.cpp`](https://github.com/vagrillo/llama.cpp) (branch
`moe-expansion`) che implementa l'espansione runtime degli esperti routed.

> 📘 **Primo utilizzo con una GPU da 16 GB?** Leggi **[gpu16gbguide.md](gpu16gbguide.md)** —
> guida passo passo (italiano + english) per mettere in funzione Qwen3.6-35B-A3B
> 2-bit con MoE-expansion su Windows e Linux.

```
    _                    _      _  __  __  _   _ ___
   / \   __ _  ___ _ __ | |    / \|  \/  |/ / | |_ _|
  / _ \ / _` |/ _ \ '_ \| |   / _ \ |\/| | || | || |
 / ___ \ (_| |  __/ | | | |  / ___ \ |  | | || | || |
/_/   \_\__, |\___|_| |_|_| /_/   \_\_|  |_| \_/|___|
        |___/  dedicated Qwen3.6-35B-A3B inference server
```

## Cosa fa allo startup

1. **Selezione del modello**: se esiste già un GGUF Qwen3.6-35B-A3B scaricato
   (in `$AGRILLA_MODELS_DIR`, `~/models`, `./models`, `C:\models` /
   `/mnt/c/models`), chiede all'utente quale usare; se non esiste ancora nulla,
   propone in funzione della **VRAM della GPU rilevata** (via `nvidia-smi`) il
   quant Unsloth più grande che entra in memoria (~92% della VRAM come budget),
   e lo scarica con la CLI `hf` dopo conferma. Il default con `invio` è sempre
   il consigliato; `d` = scarica un altro quant, `x` = annulla.
2. **Profilo MoE-expansion di default** — esattamente quello usato nei
   benchmark con **Qwen3.6-35B-A3B Q8_0** ( GPQA-Diamond: **84.34%**
   con espansione vs 81.82% nativo top-8):

   | parametro | valore |
   |---|---|
   | `--moe-experts` | 20 (model default 8) |
   | `--moe-expert-threshold` | 0.80 (adaptive: 5..20 esperti/token) |
   | `--moe-expert-layer-start` | 25 |
   | `--moe-expert-layer-end` | 39 (su 40 livelli) |
   | decay / renorm | 0.50 / auto (default del fork) |

   Il profilo viene iniettato **solo se l'utente non passa nessun flag
   `--moe-*` / `--q35-*`**; `--no-moe-expansion` (o `AGRILLA_NO_MOE_EXPANSION=1`)
   lo disattiva del tutto.
3. **Endpoint di default `127.0.0.1:8071`** e **apertura automatica del
   browser** (su Windows via `ShellExecute`, su Linux/WSL via `wslview`) appena
   il server è in ascolto. `--no-browser` o `AGRILLA_NO_BROWSER=1` disattivano
   l'apertura.
4. Altri default iniettati solo se assenti: `--jinja` (template chat Qwen),
   `-c 142768` (~140k di contesto) e `--parallel 4` (4 slot da 35840 token,
   la stessa configurazione dei benchmark RUN1209/Q2). Su GPU con poca VRAM
   `fit_params` del fork riduce automaticamente il contesto per farci stare
   modello + KV cache.

Ogni altro flag di `llama-server` passa direttamente al server sottostante:
contesto, temperatura, concorrenza e sampling si impostano con i flag
standard, es. `-c 8192`, `--temp 0.6`, `--top-p 0.95`, `-np 8`, `--threads 8`
(`--help` per l'elenco completo).

### Flag dedicati AgrillaMoE

| flag / env | effetto |
|---|---|
| `--agrilla-list-models` | elenca modelli locali + consigliato VRAM, ed esce |
| `--agrilla-models-dir DIR` | aggiunge una cartella di ricerca modelli |
| `--agrilla-yes` / `AGRILLA_YES=1` | conferma automatica dei download |
| `--no-browser` / `AGRILLA_NO_BROWSER=1` | non aprire il browser |
| `--no-moe-expansion` / `AGRILLA_NO_MOE_EXPANSION=1` | routing nativo top-8 |
| `AGRILLA_MODELS_DIR` | lista di cartelle (`:` su Linux, `;` su Windows) |
| `AGRILLA_REASONING_BUDGET` | inietta `--reasoning-budget N` (es. `8192`; `-1` illimitato, `0` chiude subito il pensiero) |
| `AGRILLA_REASONING` | inietta `--reasoning on\|off\|auto` (es. `off` disabilita del tutto il pensiero) |
| `--agrilla-streaming` / `AGRILLA_STREAMING=1` | **modalità streaming** (vedi sotto) per GPU con poca VRAM |

### Modalità streaming (`--agrilla-streaming`) — GPU con poca VRAM

Ispirata al progetto DS4 di antirez (streaming dei pesi con finestra in
memoria): pensata per GPU come una GTX 1050 4GB con un quant piccolo
(UD-IQ1_M ~9.4 GB) su disco. Allo startup AgrillaMoE configura:

- `--cpu-moe`: i pesi degli **esperti** restano su disco e arrivano via mmap;
  a ogni token `mul_mat_id` calcola **solo gli esperti scelti dal router**
  (la selezione statistica la fa già il modello)
- `-ngl 99`: attenzione, norme e KV cache vanno in GPU (ridotti da `fit_params`
  se la VRAM non basta)
- contesto compatto (8192, 1 slot) se non diversamente specificato
- un thread di **prefetch sequenziale** legge il GGUF in anticipo portando in
  cache RAM i layer successivi (il layout GGUF è sequenziale per layer), così
  i page fault non aspettano l'SSD

Su Windows c'è il launcher pronto `agrillamoe-lite.cmd` (usa `D:\models`).
La predizione **per esperte** dei layer successivi (modello previsionale
addestrato su log JSONL delle attivazioni) è descritta in `moe-predict.md`.

### Parametri di generazione (flag nativi di llama-server, tutti passano)

- **Contesto/concorrenza**: `-c 142768` (default), `-np 4` (default), es. `-c 65536 -np 2`
- **Reasoning**: `--reasoning on|off|auto` — `off` **disabilita il pensiero**; `--reasoning-budget N` — `-1` illimitato (default), `0` chiude subito il pensiero, `N>0` limite in token di thinking; `--reasoning-budget-message "..."` messaggio iniettato a fine pensiero
- **Temperatura/sampling**: `--temp 0.6`, `--top-p 0.95`, `--top-k`, `--min-p`, `--repeat-penalty`...
- Il riepilogo `[AgrillaMoE] avvio llama-server con:` mostra i valori effettivi di contesto, slot, stato e budget del reasoning.

### Suggerimento VRAM (catalogo unsloth/Qwen3.6-35B-A3B-GGUF)

| VRAM | quant proposto | dimensione |
|---|---|---|
| 8 GB | UD-IQ1_M | ~9.4 GB |
| 12 GB | UD-Q2_K_XL | ~11.4 GB |
| 16 GB | UD-Q3_K_XL | ~15.7 GB |
| 24 GB | UD-Q5_K_M | ~24.6 GB |
| 32 GB | UD-Q6_K_XL | ~29.7 GB |
| 40+ GB | Q8_0 (riferimento benchmark) | ~34.4 GB |

Se nessun quant entra in VRAM viene proposto il più piccolo (UD-IQ1_M) con
offload CPU parziale.

## Build

Il sorgente llama.cpp non è incluso: serve il fork branch `moe-expansion`
(cercato in `../repo` oppure `./llama.cpp` creato da `bootstrap-llama.sh`, o
forzato con `-DAGRILLA_LLAMA_DIR=...`).

### Linux (staticamente linkato, CUDA 12+)

```bash
./bootstrap-llama.sh        # solo la prima volta, se ../repo non esiste
./build-linux.sh             # output: dist/linux/agrillamoe
```

Variabili: `AGRILLA_CUDA_ARCH` (default `native`; es. `61`, `70;80;86`),
`AGRILLA_NATIVE` (`1` default = `-march=native` dell'host di build; `0` = CPU
baseline portabile, per binari da ridistribuire), `AGRILLA_BUILD_DIR`,
`AGRILLA_JOBS`. Requisiti: cmake ≥ 3.24, gcc, CUDA toolkit 12+.

**Binario portabile universale** (ridistribuibile, tutte le NVIDIA ≥8 GB da
RTX 20xx in poi più Pascal/Volta):

```bash
AGRILLA_NATIVE=0 AGRILLA_CUDA_ARCH="61;70;75;80;86;89;120" ./build-linux.sh
```

(sm_120 = RTX 50xx richiede CUDA toolkit ≥ 12.8; l'ultima arch elencata viene
inclusa anche come PTX, quindi GPU piu' nuove funzionano via JIT del driver.)

### Windows (staticamente linkato, CUDA 12+)

```powershell
powershell -ExecutionPolicy Bypass -File build-windows.ps1   # output: dist\windows\agrillamoe.exe
# portabile multi-arch:
powershell -ExecutionPolicy Bypass -File build-windows.ps1 -CudaArch "61;75;86;89" -Native 0 -Jobs 3
```

Parametri: `-CudaArch` (default `native`), `-Native` (default 1), `-Jobs`
(default 4), `-BuildDir`. Requisiti: Visual Studio 2019 BuildTools (VC++ +
CMake/Ninja bundled) e CUDA toolkit 12.x. Per RTX 50xx (sm_120) servono CUDA
toolkit ≥ 12.8: `-CudaArch "61;75;86;89;120"`.

**Copertura GPU dei binari di release**: Linux = sm 61, 70, 75, 80, 86, 89,
120 + PTX (GTX 10xx, V100, RTX 20xx/30xx/40xx/50xx, A100); Windows = sm 61,
75, 86, 89 + PTX 89 (GTX 10xx, RTX 20xx/30xx/40xx; RTX 50xx via ricompilazione
con CUDA ≥ 12.8).

> Nota: non usare su una GPU un binario compilato solo per un'altra
> architettura — anche se il PTX permette l'avvio, i kernel possono crashare
> alla prima generazione (dispatch runtime vs guardie di compilazione). I
> binari di release includono i cubin reali per tutte le architeture elencate.

## GPU AMD (es. Radeon RX 7800 XT) e altre

AgrillaMoE gira anche su AMD con il **backend Vulkan** (e, ricompilando, con
**HIP/ROCm**). La MoE-expansion è logica di routing nel grafo: funziona
uguale su tutti i backend (CUDA, Vulkan, ROCm, CPU).

- **Vulkan (consigliato, funziona subito)**: build con backend Vulkan
  abbinato a CUDA, stesso eseguibile per NVIDIA e AMD; su macchina AMD il
  server usa la GPU via Vulkan (RDNA3 inclusa, RX 7800 XT = gfx1101). Serve
  solo il driver con Vulkan runtime (sempre presente coi driver AMD/NVIDIA).
  Build Windows: `build-windows.ps1 -Vulkan 1 ...`; Linux:
  `AGRILLA_VULKAN=1 ./build-linux.sh`. All'avvio `--device Vulkan0` forza la
  GPU Vulkan quando ce ne sono più di una (`--list-devices` per l'elenco).
- **HIP/ROCm (prestazioni migliori su AMD, build dedicata)**: serve AMD HIP
  SDK/ROCm; Windows: `build-windows.ps1 -Hip 1 -AmdTargets gfx1101`;
  Linux: `AGRILLA_HIP=1 AGRILLA_AMD_TARGETS=gfx1101 ./build-linux.sh`
  (RX 7800 XT = `gfx1101`; `gfx1100` = 7900 XTX/XT).

Stato test: NVIDIA testato direttamente (CUDA e Vulkan); **AMD non testato su

## macOS (Apple Silicon, backend Metal)

Su Mac M1/M2/M3/M4 AgrillaMoE usa il **backend Metal** di llama.cpp: la
MoE-expansion funziona identica (routing nel grafo, `mul_mat_id` supportato da
Metal) e la **memoria unificata** semplifica tutto — il "budget" è la RAM
totale, la modalità streaming serve raramente:

| RAM Mac | quant consigliato |
|---|---|
| 16 GB | UD-IQ2_M / UD-Q2_K_XL |
| 24 GB | UD-Q4_K_XL |
| 32 GB | UD-Q6_K_XL |
| 48+ GB | Q8_0 per intero |

Build sul Mac (serve `xcode-select --install` e `brew install cmake`):

```bash
./bootstrap-llama.sh && ./build-macos.sh    # output: dist/macos/agrillamoe
```

**Binari precompilati**: la GitHub Action `build-binaries` compila su runner
Apple Silicon a ogni tag `v*` e allega `agrillamoe-…-macos-arm64-metal.tar.gz`
alla release; si può lanciare anche manualmente (tab Actions → Run workflow).

> Nota su **MLX**: AgrillaMoE è basato su llama.cpp/C++; il backend Metal è la
> via nativa su Mac e ha prestazioni comparabili a MLX su GGUF. Un port
> dell'espansione MoE su mlx-lm (Python) sarebbe un progetto separato — vedi
> `moe-predict.md` per la parte di analisi riusabile.
hardware nostro** — il percorso Vulkan è identico su tutte le GPU, ma
feedback da utenti AMD (soprattutto RX 7800 XT) è benvenuto: aprite una
issue su GitHub con GPU, driver e output di `agrillamoe --list-devices`.

## Uso

```bash
./agrillamoe                       # menu interattivo, defaults AgrillaMoE
./agrillamoe -m ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
./agrillamoe --host 0.0.0.0 --port 9000 --no-browser
./agrillamoe -c 65536 -np 2        # contesto/concorrenza custom
./agrillamoe --reasoning off       # disabilita il pensiero
./agrillamoe --reasoning-budget 8192   # limita il thinking a 8192 token
./agrillamoe --temp 0.6 --top-p 0.95
./agrillamoe --moe-experts 12 --moe-expert-threshold 0.7   # profilo custom
```

API: OpenAI-compatibili su `http://127.0.0.1:8071/v1` (+ web UI sulla radice),
identiche a `llama-server`; include anche l'endpoint **Anthropic
`/v1/messages`**, quindi si possono collegare direttamente **Claude Code** e
gli agent OpenAI-compatibili (Aider, Cline, OpenCode, Continue, Zed, Goose) —
istruzioni passo passo in [gpu16gbguide.md](gpu16gbguide.md), sezione 8.

## Struttura

```
AgrillaMoE/
├── CMakeLists.txt        # progetto standalone che aggiunge il fork come subdirectory
├── src/main.cpp          # banner, selezione modello/VRAM, download hf, default, browser
├── build-linux.sh        # build Linux statico CUDA
├── build-windows.ps1     # build Windows statico CUDA (VS2019 BuildTools + CUDA 12)
├── bootstrap-llama.sh    # clone del fork moe-expansion se serve
└── dist/                 # binari prodotti (non versionati)
```

## Risultati benchmark: HumanEval

Prima misura pubblica di HumanEval per Qwen3.6-35B-A3B — A/B expansion vs
stock routing su RTX 2080 Ti con UD-IQ4_XS 4-bit (thinking budget 4096):

| Config | pass@1 | Decode |
|---|---|---|
| Stock top-8 | 89,63% (147/164) | 69,3 tok/s |
| MoE-expansion 20 | **90,85%** (149/164) | 56,0 tok/s |

Dettagli completi (metodologia, analisi appaiata, caveats) in
**[humaneval-eval.md](humaneval-eval.md)**.

## Contesto

- Espansione MoE: routing runtime con più esperti routed del top-K nativo
  (vedi `docs/moe-expansion.md` nel fork).
- Benchmark di riferimento (RUN1209, vast.ai V100): Qwen3.6-35B-A3B **Q8_0**
  con `N=16 T=0.8 L25–39` → GPQA-Diamond 84.34% vs 81.82% nativo (+2.5 punti).
  Questo è il profilo iniettato di default da AgrillaMoE.

## Licenza

MIT (come llama.cpp). Questo progetto include/modifica codice di
[llama.cpp](https://github.com/ggml-org/llama.cpp) e del fork
`vagrillo/llama.cpp` (branch moe-expansion).
