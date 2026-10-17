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

### Findings

1. **Il raddoppio conservativo (4→8) sui layer tardivi è il sweet spot**: +8 punti
   GPQA-Diamond per -6% di throughput. Stesso pattern del profilo Qwen3.6 (8→20),
   qui amplificato dalla larghezza nativa più stretta (top-4).
2. **L'espansione aggressiva danneggia**: e12 su layer 20-39 perde 6 punti rispetto
   allo stock. La soglia 0.80 tiene pochi esperti extra, ma il mix pesato con tanti
   rank deboli in decay diluisce il routing. Su un budget attivo di 2B, oltre il
   raddoppio si peggiora.
3. **Il budget 8K è un vincolo attivo**: 40-52% delle domande satura il tetto
   (dipende dalla config). La config vincitrice è anche quella che ragiona in
   modo più efficiente (meno cap-hit, mediano più basso); la peggiore è quella
   che satura di più — più esperti deboli → più pensiero circolare.

## Run a budget 16K (in corso)

GPQA-Diamond completo (198 domande), budget 16384, ctx 81920 (4 slot × 20480),
config **noexp** e **exp01**. Risultati in aggiornamento.

---
Setup VM: vast.ai V100-SXM2-16GB, driver 580, CUDA 12.8. Date: 2026-10-17.
