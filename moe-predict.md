# MoE expert prediction from activation logs

Design document for **per-expert** prefetching in streaming mode
(`--agrilla-streaming`), after antirez's DS4: weights live on disk, a RAM
window acts as staging and VRAM holds what the current step needs. This
document describes how to go from per-layer sequential prefetch (already
implemented) to **early expert selection** based on statistical patterns
accumulated across previous runs.

## 1. The data: per-token, per-layer JSONL logs

The moe-expansion fork already exposes a per-layer observer
(`res->moe_expert_counts`, see the `moe_expand` branch of
`llama-graph.cpp:build_ffn`): at each ubatch it reads back `sel_count` (how
many experts were kept). Prediction needs more: **which** experts, per token.
The information already exists in the graph:

- `selected_experts` — `[n_used, n_tokens]` I32 tensor: candidate IDs (after
  top-K/expansion, before the threshold cut)
- post-expansion `weights` — `[1, n_used, n_tokens]` F32: weight>0 ⇒ expert
  kept (the adaptive cut zeroes the others)

**Implemented extension** (`llama-graph.cpp` + `llama-context.cpp`): when the
`LLAMA_MOE_EXPERT_LOG=<file.jsonl>` environment variable is set, both tensors
are marked as graph outputs for every expanded layer (the same
`ggml_set_output` mechanism already used for `sel_count`) and, in the
post-compute readback, one JSON line is written per ubatch:

```json
{"pos0":1234,"n":3,
 "layer":{"25":[[41,907,12],[41,907,12,333],[41,12]],"26":[[...],...]}
}
```

i.e. for every token, the list of **kept experts** (weight>ε) layer by layer,
expansion included. Cost: ~1–2 KB/token at 16 experts over 15 layers; logging
is off by default, with `LLAMA_MOE_EXPERT_LOG_EVERY=N` sampling to reduce
volume.

## 2. The analysis: `moe_predict.py`

Already in the repo. From a `run.jsonl` it produces:

1. **per-layer frequencies** and top-M coverage: how many distinct experts
   cover 50/80/95% of activations → the size of the per-layer "hot window"
2. **Markov transitions L→L+1** (same token): `P(e_{L+1} | e_L)` as sparse
   counts; the conditioned top-M coverage says how much it's worth predicting
   the next layer from the current one
3. **persistence**: how often the expert at token t repeats the one at token
   t−1 in the same layer

Output: `predictions.json` (per layer: `top_freq`, top-M successors for
frequent experts) directly consumable by a prefetcher.

`moe_predict_set.py` goes further: **set-conditioned prediction** with a
train/test split — given the *entire* active set of layer L, the predicted
set for L+1 is the union of the top-k successors of each source expert,
capped at a byte budget. This is the metric a real prefetcher would achieve.

## 3. The long-session harness: `moe-predict-run.sh`

One command for a vast.ai box (or any Linux GPU machine): builds AgrillaMoE
natively, downloads the model and a HuggingFace prompt dataset (default
`HuggingFaceH4/no_robots`), runs a long query session with the expert logger
enabled, then produces the statistics:

```bash
git clone https://github.com/vagrillo/AgrillaMoE && cd AgrillaMoE
MOE_N_PROMPTS=200 MOE_MAX_TOKENS=512 bash moe-predict-run.sh
# output: run.jsonl, moe-report.txt, predictions.json
```

## 4. Phase 2: the prefetchers (implemented and measured)

Two prefetchers were implemented in the fork and measured:

**Temporal** (`LLAMA_MOE_EXPERT_PREFETCH=1`): after every decode token, the
bytes of the experts just used are moved into VRAM ahead of time
(`cudaMemPrefetchAsync`, unified memory) — betting on token-to-token routing
persistence. v2 adds a **static hot set**: in-memory per-layer frequency
counters, with the top-N experts per layer periodically re-prefetched
(`LLAMA_MOE_PREFETCH_STATIC`, `LLAMA_MOE_PREFETCH_STATIC_EVERY`).

**Chunked decode** (`--agrilla-chunk-predict`, DS4-style): the sched's eval
callback observes the routing tensors at chunk boundaries → backend sync →
real routing is read → the next chunk's candidates are predicted with the
online Markov chain → prefetched while the current chunk computes.

### Measured results

**V100 16GB, Qwen3.6-35B Q8_0 (34 GB), unified memory:**

| Config | decode | Δ |
|---|---|---|
| gpu-streaming baseline | 4.18 t/s | — |
| + temporal prefetch (v1) | 4.56 t/s | +9% |
| + static+temporal (v2) | 4.46 t/s | +7% |

**RTX 2080 Ti 22GB, Q8_0:**

| Config | decode | Δ |
|---|---|---|
| gpu-streaming baseline | 5.19 t/s | — |
| + chunked, budget 24 | 4.46 t/s | **−14%** |
| + chunked, budget 12, boundaries 26/30/34/38 | 4.88 t/s | −6% |

### Experimental verdict

Chunked decode **does not pay off** on these configurations, for two measured
reasons:

1. With 22GB VRAM, 64% of the model is already resident and the driver's LRU
   does what a static prediction would have done — the residual margin is the
   36% of misses, and every prefetched byte (including predicted-but-unused
   waste) competes with faults on an already-saturated PCIe x8 bus
2. Halving the budget (24→12) recovers half the damage, confirming the problem
   is prefetch traffic, not prediction quality

The 47% cross-layer signal is real but **already captured by the driver**: the
prediction is calibrated on the same routing distributions the managed-page
LRU tracks, at finer granularity. A possible phase 3 would target only the
regime where the driver fails: per-token working set ≫ VRAM (models ≥4× VRAM)
on an unsaturated bus — or selective prefetch in CPU execution, where the
"bus" is the DDR4-CPU channel and expert placement in large contiguous pages
would make kernel readahead effective. The infrastructure (logger, callback,
prefetchers, measurements) stays in the fork as the base for that work.

---

# Versione italiana (Italian version)

*Same content as the English version; the cited experiments are the same.*

# Predizione degli esperti MoE dal log delle attivazioni

Documento di design per il prefetch **per esperto** nella modalità streaming
(`--agrilla-streaming`), sul modello di DS4 (antirez): i pesi vivono su disco,
una finestra in RAM fa da staging e la VRAM tiene ciò che serve al passo
corrente. Qui descriviamo come passare dal prefetch sequenziale per layer
(già implementato) alla **selezione anticipata degli esperti** sulla base di
pattern statistici accumulati dai run precedenti.

## 1. Il dato: log JSONL per token, layer per layer

Il fork moe-expansion espone già un observer per layer
(`res->moe_expert_counts`, vedi `llama-graph.cpp:build_ffn` rami `moe_expand`):
a ogni ubatch legge `sel_count` (quanti esperti tenuti). Per la predizione
serve di più: **quali** esperti, per ogni token. Le informazioni esistono già
nel grafo:

- `selected_experts` — tensore `[n_used, n_tokens]` I32: gli ID dei candidati
  (dopo top-K/espansione, prima del taglio di soglia)
- `weights` post-espansione — `[1, n_used, n_tokens]` F32: peso>0 ⇒ esperto
  tenuto (il taglio adattivo azzera gli altri)

**Estensione proposta** (`llama-graph.cpp` + `llama-context.cpp`): quando è
attiva la variabile d'ambiente `LLAMA_MOE_EXPERT_LOG=<file.jsonl>`, marcare
come output anche questi due tensori per ogni layer espanso (stesso meccanismo
di `ggml_set_output` già usato per `sel_count`) e, nel readback post-compute
(dove oggi si accumulano le medie), scrivere una riga JSON per ubatch:

```json
{"pos":[1234,1235,1236],
 "layer":{"25":[[41,907,12],[41,907,12,333],[41,12]],"26":[[...],...]}
}
```

cioè per ogni token l'elenco degli **esperti tenuti** (peso>ε) layer per
layer, espansione inclusa. Costi: ~1–2 KB/token a 16 esperti su 15 layer;
il log si attiva solo quando serve (default off), con campionamento
`LLAMA_MOE_EXPERT_LOG_EVERY=N` per ridurre il volume.

## 2. L'analisi: `moe_predict.py`

Già pronto nel repo. Da un `run.jsonl` produce:

1. **frequenze per layer** e copertura dei top-M: quanti esperti distinti
   servono per coprire il 50/80/95% delle attivazioni → la dimensione della
   "finestra calda" per layer
2. **transizioni Markov L→L+1** (stesso token): `P(e_{L+1} | e_L)` come
   conteggi sparsi; la copertura top-M condizionata dice quanto vale
   prevedere il layer successivo da quello corrente
3. **persistenza**: quanto l'esperto del token t ripete quello del token t−1
   nello stesso layer (nei MoE tende a essere alto: trace di routing stabile)

Output: `predictions.json` (per layer: `top_freq`, successori top-M per
esperte frequenti) direttamente consumabile da un prefetcher.

## 3. Il prefetcher predittivo (fase 2)

Con `predictions.json` + gli offset dei tensori nel GGUF (il parser già
presente in AgrillaMoE va esteso alla sezione tensori: nome, dims, type,
offset) si può precaricare **per intervallo di byte**:

1. al layer L, il router ha appena scelto gli esperti E_L
2. il prefetcher consulta le transizioni e lancia letture asincrone
   (Windows: `PrefetchVirtualMemory`; Linux: `readahead()`/`posix_fadvise`)
   degli slot di `ffn_{gate,up,down}_exps` degli esperti più probabili di
   L+1..L+k, solo se non già in cache (bit map degli slot residenti)
3. gli slot sono contigui per esperto nei tensori GGUF ⇒ letture
   grandi-e-poche, amichevoli con l'SSD

Nota architetturale onesta: il router decide dentro lo stesso forward pass,
quindi la predizione non può precedere la scelta *dentro* il layer corrente;
guadagna però l'intera durata del compute dei layer L..L+k-1, che a 4 GB di
VRAM e compute CPU è decine di millisecondi per layer — più che sufficienti
per nascondere la latenza SSD delle righe predette. Il guadagno reale è
misurabile come rapporto (hit-rate × tempo-SSD-risparmiato); moe_predict.py
stampa già l'hit-rate atteso prima di scrivere una riga di C++.

## 4. Stato

| Pezzo | Stato |
|---|---|
| streaming per layer (`--cpu-moe` + prefetch sequenziale + ctx compatto) | ✅ implementato in AgrillaMoE |
| logger JSONL per-token nel fork | ✅ implementato (`LLAMA_MOE_EXPERT_LOG=file.jsonl` nel fork) |
| analisi/statistica + tabella predizioni (`moe_predict.py`) | ✅ pronto |
| prefetch per esperto su offset GGUF | ⏭ fase 2, guidata dai numeri sotto |

## Numeri reali (prima raccolta: 591 token, 5 query, IQ1_M, espansione 20)

- routing sano: ~255/256 esperti distinti usati per layer
- copertura statica top-20 per layer: **25-49%** (media ~38%)
- predizione markoviana L→L+1 (condizionata al top-1 del layer): **31-67%**
  (molte transizioni oltre il 50%)
- sessioni lunghe su dataset vario (moe-predict-run.sh) e condizionamento
  sull'INTERO insieme di esperti del layer L (non solo il top-1) sono i due
  leve per avvicinarsi al 75%+ che servirebbe al prefetch selettivo

## Sessione lunga su vast.ai (GPU grossa)

```bash
git clone https://github.com/vagrillo/AgrillaMoE && cd AgrillaMoE
MOE_N_PROMPTS=200 MOE_MAX_TOKENS=512 bash moe-predict-run.sh
# output: run.jsonl, moe-report.txt, predictions.json
```
| analisi/statistica + tabella predizioni (`moe_predict.py`) | ✅ pronto |
| prefetch per esperto su offset GGUF | ✅ implementato e misurato (vedi sotto) |

## Risultati sperimentali fase 2 (V100 16GB, Qwen3.6-35B Q8_0 34GB, unified memory)

Prefetcher implementato nel fork: `LLAMA_MOE_EXPERT_PREFETCH=1` — dopo ogni
token di decode, anticipa in VRAM (cudaMemPrefetchAsync) i byte degli esperti
appena usati (temporale) e periodicamente i top-N per frequenza (statica, v2).

| Configurazione | decode | Δ |
|---|---|---|
| baseline gpu-streaming | 4,18 t/s | — |
| + prefetch temporale (v1) | 4,56 t/s | +9% |
| + statica+temporale (v2) | 4,46 t/s | +7% |

**Perché così poco, a fronte del 47% misurato?** Il segnale forte del 47% è
**trasversale tra layer entro lo stesso token** (il router di L+1 correla con
quello di L dello stesso passaggio). Ma llama.cpp esegue l'intero grafo del
token in una volta: al momento del prefetch (tra due token) non conosciamo il
routing del token successivo, e la persistenza temporale dello *stesso* layer
tra token consecutivi è debole (~10%) — che è esattamente il +9% misurato.
Il driver LRU di CUDA gestisce per conto suo parte del riuso (baseline 4,18
t/s supera il limite naive senza cache di ~3,3 t/s).

**Chunked decode (implementato, `--agrilla-chunk-predict`)**: eval callback
dello sched → sync ai confini di chunk → lettura routing reale → predizione
Markov online → prefetch candidati del chunk successivo. Misurato:

| Configurazione | decode | Δ |
|---|---|---|
| **V100 16GB** + Q8_0: baseline | 4,18 t/s | — |
| V100: + prefetch temporale | 4,56 t/s | +9% |
| **RTX 2080 Ti 22GB** + Q8_0: baseline | 5,19 t/s | — |
| 2080 Ti: + chunked budget 24 | 4,46 t/s | **−14%** |
| 2080 Ti: + chunked budget 12, confini 26,30,34,38 | 4,88 t/s | −6% |

**Verdetto sperimentale**: il chunked decode *non* paga su queste configurazioni.
Due motivi misurati: (1) con 22GB di VRAM il 64% del modello è già residente e
il driver LRU fa quello che la predizione statica avrebbe fatto — il margine
residuo è il 36% di miss, e ogni byte di prefetch (incluso lo spreco
predetto-ma-non-usato) compete con i fault su un bus PCIe x8 già saturo;
(2) la riduzione del budget (24→12) recupera metà del danno, confermando che
il problema è il traffico di prefetch, non la qualità della predizione.

Il segnale del 47% resta reale ma è **già incassato dal driver**: la previsione
è calibrata sulle distribuzioni di routing, e il LRU delle pagine managed
insegue le stesse distribuzioni con granularità fine. Un'eventuale fase 3
dovrebbe attaccare il solo scenario in cui il driver fallisce: working set per
token >> VRAM (modelli ≥4× la VRAM) e bus non saturo — oppure il prefetch
selettivo nell'esecuzione CPU (dove il "bus" è il canale DDR4-CPU e il
*collocazione* degli esperti in pagine enormi contigue renderebbe il readahead
del kernel efficace). L'infrastruttura (logger, callback, prefetcher,
misurazioni) resta nel fork come base per quel lavoro.
