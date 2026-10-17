# Benchmark: LiquidAI LFM2.5-24B-A2B su AgrillaMoE

Prima caratterizzazione del profilo MoE-expansion su **LFM2.5-24B-A2B**
(arch `lfm2moe`: 40 layer, 64 esperti, **top-4 nativo**, ~2B attivi),
sulla falsariga del protocollo usato per Qwen3.6-35B-A3B.

## Compatibilità

| componente | stato |
|---|---|
| fork llama.cpp (branch `moe-expansion`) | ✅ nessuna modifica necessaria: il grafo `lfm2` chiama `build_moe_ffn` con gating softmax e `norm_w=true`, tutte le guardie di expansion/reduce passano |
| wrapper AgrillaMoE | ✅ firma dedicata `bc==40 && ec==64 && eu==4` → profilo auto-iniettato **esperti 8, soglia 0.80, layer 25-39** (prima della firma Qwen3.6, che condivide `bc==40`) |
| `--moe-reduce` / `--moe-reduce-layers` | ✅ validato (rinormalizzazione massa-preservante) |
| immagine serverless Docker | ✅ `ghcr.io/vagrillo/agrillamoe-serverless` — il profilo auto si applica anche in serverless |

Validazione funzionale su LFM2-8B-A1B (stesso arch, top-4 nativo): stock,
`--moe-reduce 0.5` ed expansion 4→20 generano correttamente.

## Risultati GPQA-Diamond @ reasoning budget 8K

Setup: Tesla V100-16GB, QAD-Q4_0 (13.6 GB, full GPU), ctx 65536 (4 slot × 16384),
KV q8_0, `--reasoning-budget 8192`, temperatura 0, evalscope, batch 4,
**50 domande** (stesso set per tutte le config, confronto accoppiato).

| config | accuratezza | throughput | tpot medio | domande al tetto 8K |
|---|---|---|---|---|
| noexp (stock top-4) | 70.0% | 38.7 tok/s | 25.2 ms | 23/50 (46%) |
| **exp01: e8, thr 0.80, layer 25-39** | **78.0%** | 36.5 tok/s | 27.0 ms | 21/50 (42%) |
| exp02: e12, thr 0.80, layer 20-39 | 64.0% | 39.2 tok/s | 25.2 ms | 26/50 (52%) |

### Confronto con il BF16 ufficiale

LiquidAI riporta per LFM2.5-24B-A2B in BF16 **74.42%** su GPQA Diamond
([fonte: HF blog LiquidAI/lfm25-24b-a2b](https://huggingface.co/blog/LiquidAI/lfm25-24b-a2b),
valutato con vLLM e i parametri di generazione raccomandati da ciascun benchmark,
senza budget di reasoning esplicito). Per contesto, nello stesso confronto:
Qwen3.6-35B-A3B 83.69, Gemma-4-26B-A4B 74.90.

| run (QAD-Q4_0 4-bit, budget 8K) | GPQA-Diamond | vs BF16 74.42% |
|---|---|---|
| noexp (stock top-4) | 70.0% | −4.4 |
| **exp01 (e8, layer 25-39)** | **78.0%** | **+3.6** |

**L'expansion e8 non solo recupera il costo della quantizzazione 4-bit, supera
il riferimento BF16 ufficiale.** Attenzione al confronto: i protocolli differiscono
(vLLM vs llama.cpp, 50 domande vs set completo, budget 8K vs non specificato,
parametri di sampling diversi) — il confronto è indicativo, non apples-to-apples.

### Findings

1. **Il raddoppio conservativo (4→8) sui layer tardivi è il sweet spot**: +8 punti
   GPQA-Diamond per -6% di throughput, portando il 4-bit sopra il riferimento
   BF16 ufficiale. Stesso pattern del profilo Qwen3.6 (8→20),
   qui amplificato dalla larghezza nativa più stretta (top-4).
2. **L'espansione aggressiva danneggia**: e12 su layer 20-39 perde 6 punti rispetto
   allo stock. La soglia 0.80 tiene pochi esperti extra, ma il mix pesato con tanti
   rank deboli in decay diluisce il routing. Su un budget attivo di 2B, oltre il
   raddoppio si peggiora.
3. **Il budget 8K è un vincolo attivo**: 40-52% delle domande satura il tetto
   (dipende dalla config). La config vincitrice è anche quella che ragiona in
   modo più efficiente (meno cap-hit, mediano più basso); la peggiore è quella
   che satura di più — più esperti deboli → più pensiero circolare.

## Run a budget 16K (completo, 198 domande)

GPQA-Diamond completo, budget 16384, ctx 81920 (4 slot × 20480), config
**noexp** e **exp01**. Confronto accoppiato (le 50 domande dell'8K sono un
prefisso del set 198, stessi indici):

| config | paired-50 @8K | paired-50 @16K | full-198 @16K |
|---|---|---|---|
| noexp (stock top-4) | 70.0% | **80.0%** | 73.2% |
| exp01 (e8, L25-39) | 78.0% | 74.0% | 71.7% |

Segmento accoppiato "ex-cap-8K" (domande che a 8K saturavano il budget):

| segmento | @8K | @16K |
|---|---|---|
| noexp, 23 domande ex-cap | 34.8% | **56.5%** |
| exp01, 21 domande ex-cap | 66.7% | 52.4% |

**Inversione col budget**: a 8K l'expansion vince (+8), a 16K perde (−1.5 su
full, −6 su paired) — il routing espanso converge più sicuro sotto pressione ma
overthinking quando ha spazio (le sue risposte ex-cap quasi raddoppiano in
lunghezza e peggiorano). Il budget 16K fa salire lo stock sopra il riferimento
BF16 (80.0% paired / 73.2% full vs 74.42%). L'expansion resta il modo di
raggiungere la stessa classe di punteggio con ~metà budget di reasoning
(78.0% @8K, ~6.9K tok/risposta vs 80.0% @16K, ~9.7K tok/risposta).
Dettagli completi: [lfm25-expansion-gpqa.md](lfm25-expansion-gpqa.md).

---
Setup VM: vast.ai V100-SXM2-16GB, driver 580, CUDA 12.8. Date: 2026-10-17.
