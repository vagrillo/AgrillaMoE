# Expert Rank — Qwen3.6-35B-A3B (fork moe-expansion)

Ranking di utilizzo degli esperti MoE per la quantizzazione mixed-precision,
raccolto su VM vast.ai (Tesla V100-PCIE-32GB) con workload GPQA-Diamond.

## File di questa directory

| File | Contenuto |
|---|---|
| `expert-usage-total.json` | Conteggi cumulativi per (layer, esperto) raccolti durante il run — 441.745 token, 20.591.738 selezioni |
| `expert-order.txt` | Per ogni MoE layer: i 256 esperti ordinati per uso decrescente — input di `llama-quantize --split-experts` |
| `quantization-plan.json` | Piano per-esperto: per ogni layer i 256 esperti ordinati per uso con fascia di quantizzazione assegnata |
| `make_plan.py` | Generatore del piano dal JSON d'uso (ricalcolabile) |
| `make_mixed_gguf.py` | Repack lossless dei tensori esperti Q8_0 in 4 pool per layer (alternativa manuale al `--split-experts` del fork) |
| `hf-modelcard.md` | Model card di [vagrillo/Qwen35B-A4B-MOExpMixedQuant](https://huggingface.co/vagrillo/Qwen35B-A4B-MOExpMixedQuant) |
| `tools/` | Verifica/patch delle tabelle pool nel GGUF e test di regressione dei percorsi `mul_mat_id` |

## Sessione 2026-10-25: fix e validazione del GGUF mixed

Primo run completo del modello mixed (21.5 GB) su V100. Esito:

1. **Crash assert** `ids_to_sorted_host.size() == ne_get_rows` in
   `ggml_cuda_mul_mat_id` al primo prefill > 64 token: il fallback sorted
   assumeva un solo slot per (token, esperto), falso con le pool. Fix nel
   fork (compazione per coppia, commit `744a55ac1`), compresa la variante
   MMQ scatter già coperta da `mm_ids_helper` per-pair (`56f65c961`).
2. **Output degenere** (eco del prompt, risposte vuote) a prescindere da
   espansione e backend: il quantizzatore `--split-experts` scriveva le
   tabelle `ffn_exps_local/mask` leggendo `order[256+pos]` — l'ordine del
   layer successivo (e oltre la fine del buffer per l'ultimo layer).
   Fix nel fork (commit `48872dc79`); il GGUF già prodotto è stato
   corretto patchando in-place le 320 tabelle (40 layer × 8) da
   `expert-order.txt` con `tools/patch_tables.py` (verifica byte-a-byte
   prima di ogni scrittura, backup in `table-backup.npz`, verifica con
   `tools/check_tables.py`).
3. **Validazione post-fix**: server `--moe-experts 20 --moe-expert-threshold
   0.95` su V100 — risposte coerenti con e senza espansione, su tutti i
   percorsi dispatch (MMVQ ≤ 8 token, MMQ scatter 9–64, fallback sorted
   > 64: `tools/test-mixed-fix.sh`); memcheck compute-sanitizer pulito
   sui tre percorsi (`tools/test-san-fix.sh`).

I GGUF richiesti dal modello HF sono stati rigenerati/patchati con le
tabelle corrette: rigenerando da Q8_0 serve il fork ≥ `48872dc79`.

## Modello usato durante il ranking

| Campo | Valore |
|---|---|
| Modello | **Qwen3.6-35B-A3B** (MoE, arch `qwen35moe`) |
| GGUF a runtime | [`unsloth/Qwen3.6-35B-A3B-GGUF` → `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (17.7 GB, schema dinamico Unsloth: esperti ~IQ3_S/IQ4_XS, attention Q8_0, shared expert Q8_0) |
| Sorgente per la requant | [`Qwen3.6-35B-A3B-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (36.9 GB, stesso repo) + `imatrix_unsloth.gguf_file` |
| Architettura | 40 layer MoE × **256 esperti** (top-8 nativo) + 1 **shared expert** per layer (sempre attivo, gate sigmoid — NON incluso nei conteggi: non passa dal router) |
| Runtime | fork [`vagrillo/llama.cpp`](https://github.com/vagrillo/llama.cpp/tree/moe-expansion), branch `moe-expansion`, commit `f4da1de` + patch `moe-expert-usage.patch` (v2) |
| Hardware | Tesla V100-PCIE-32GB, driver 580.178.04, CUDA toolkit 12.8 (sm_70) |

## Parametri di espansione MoE (flag del fork)

```
-m Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
-ngl 999 -c 65536 -ctk q8_0 -ctv q8_0
--temp 0 --jinja --parallel 4
--reasoning-budget 8192
--reasoning-budget-message "Stop reasoning now: commit to your best-supported answer and respond with the final answer."
--moe-experts 20 --moe-expert-threshold 0.95        # su TUTTI i layer (start/end omessi)
```

Espansione attiva su layer 0..39 di 40: budget 20 candidati/token, floor 5,
mantenuti i rank con peso > 0.95 × peso del rank 10 (~11.5 esperti/token osservati).

## Metodologia di raccolta

1. **Patch `moe-expert-usage.patch`** (v2) sul fork: i tensori di routing
   (`selected_experts` I32 e `weights` F32 post-espansione) vengono esposti come
   output del grafo e letti da una **eval callback dello sched** (pattern
   `moe_chunk_cb` del fork): per ogni tensore osservato il backend si
   sincronizza e offre i dati. Il conteggio avviene lì, per (layer, esperto),
   su tutti i token di ogni ubatch.
2. **Semantica del conteggio**: un esperto è "usato" se il suo peso
   post-espansione è > 1e-6 (esperto mantenuto, contribuisce all'output del
   token). Comprendono gli 8 nativi (sempre mantenuti) + gli esperti
   dell'espansione sopra soglia. Gli esperti candidati tagliati (peso 0) e gli
   esperti fuori dal top-20 NON contano. Lo shared expert non passa dal router:
   non è nel conteggio né può esserlo nel piano.
3. **Dump**: JSON atomico (`tmp` + `rename`) ogni 256 token su
   `expert-usage-server.json`; un poller (`expert_usage_poller.py`) accumula i
   delta in `expert-usage-total.json` ogni 15 s — tabella aggiornata durante il
   run, robusta a interruzioni e riavvii del server.
4. **Workload**: replay delle 93 domande GPQA-Diamond già processate da
   evalscope (`gpqa_diamond`, prompt identici: template multiple-choice +
   "Think step by step"), `max_tokens` 8192, temperatura 0, concorrenza 4
   (batch server 4 slot), reasoning budget 8192.
5. **Validazione**: tutte le versioni precedenti del conteggio (readback a fine
   grafo) sono state scartate — perdevano gli ubatch di decode per aliasing dei
   buffer (CUDA graphs + topologie concorrenti). La v2 è validata: **40/40
   layer a 10.57–12.74 selezioni/token** (atteso ~11.5, confermato anche dalle
   statistiche interne del server `moe: experts/token avg`).
6. **Ordine d'uso**: il ranking è la semplice ordinazione decrescente dei
   conteggi cumulativi; nessuna trasformazione. `total_tokens` per layer è in
   `layer_tokens` (rate = selezioni/token per layer).

## Caveats

- **Bias di dominio**: il ranking riflette il routing su GPQA-Diamond (QA
  scientifico con reasoning lungo). Altri domini (coding, chat) possono
  spostare l'uso; per un modello general-purpose valutazioni di raccolta su
  workload misti.
- **Distribuzione quasi piatta**: nelle fasce di coda i confini (rank 16/17,
  64/65) hanno gap 0–2%, dell'ordine del rumore di campionamento: la
  classificazione della coda è indicativa, la testa (top-16) è netta
  (top-1 ~45k vs mediana ~11k selezioni).
- Concorrenza 4 e budget reasoning 8192 fanno parte del workload: cambiandoli
  cambia (leggermente) la distribuzione.

## Piano di quantizzazione (4 fasce per layer)

| Fascia | Rank esperto (per layer) | Tipo | bpw |
|---|---|---|---|
| 1 | 1–16 | Q8_0 | 8.5 |
| 2 | 17–64 | Q6_K | 6.5625 |
| 3 | 65–192 | Q4_K | 4.5 |
| 4 | 193–256 | Q2_K | 2.5625 |

Media esperti: **4.652 bpw** → ~18.7 GB per gli esperti routati + ~2.5 GB
(attention Q8_0, embedding Q8_0, output Q6_K, shared expert Q8_0, router F32,
norme) ≈ **21.2 GB totali** stimati.

## Realizzazione del GGUF

GGUF quantizza un intero tensore con un solo tipo: `blk.N.ffn_gate_up_exps` è
un unico tensore `[2·n_ff, n_embd, 256]`. Il piano per-esperto richiede quindi:

1. **Repack** dalla Q8_0: split dei tensori esperti in 4 sotto-tensori per
   layer (`_hi` 16 / `_mid` 48 / `_lo` 128 / `_tail` 64 esperti) copiando i
   blocchi Q8_0 raw (i blocchi da 32 elementi non attraversano mai il confine
   tra esperti: la dimensione esperto è l'ultima e ogni esperto è contiguo) —
   copia senza perdita.
2. **Requant per fascia** con `llama-quantize --tensor-type` su ciascun
   sotto-tensore (Q6_K/Q4_K/Q2_K dall'Q8_0 con imatrix).
3. **Estensione del fork**: il loader deve leggere i 4 sotto-tensori per layer
   e `build_moe_ffn` deve eseguire il routing su pool separati (remap/clamp
   degli id per pool, maschera pesi, somma dei parziali). Nota: la valutazione
   densa per pool moltiplica il compute delle FFN; con budget per pool basati
   sul p99 dei conteggi per token l'overhead è limitato, con valutazione
   densa pura è ~4× sulle FFN.

Alternativa senza modifiche al fork (già operativa): quantizzazione per-layer
(`llama-quantize --tensor-type` sui tensori esperti interi dei layer più caldi)
— non per-esperto, ma Zero code changes.
