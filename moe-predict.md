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
| prefetch per esperto su offset GGUF | ⏭ fase 2, dopo aver misurato hit-rate sul log reale |
