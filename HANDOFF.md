# STATO INTERMEDIO — moe-bracket e benchmark (handoff 2026-10-08)

Documento di ripresa: tutto quello che serve per continuare senza perdere lavoro.

## 1. Cosa è implementato e pushato (fork `vagrillo/llama.cpp`, branch moe-expansion)

Ultimo commit bracket: **42e8034** (Q5_K support). Il WIP-debug fprintf (77da9fd) è
ancora nel sorgente: da rimuovere prima del merge definitivo.

- **Estrazione step per-blocco** (llama-context.cpp, LLAMA_MOE_BRACKET=1): legge i
  byte di scala dei tensori esperti Q4_K/Q5_K/Q8_0/Q4_0 via backend_tensor_get,
  decodifica lo step (Q4_K/Q5_K: `get_scale_min_k4`, step = d·sc; Q8_0/Q4_0: d),
  quantizza gli step in tensori Q8_0 [n_blocks, n_out, n_exp] su CPU
  (`moe_bracket_ctx` no_alloc + malloc + `ggml_backend_cpu_buffer_from_ptr`).
- **Correzione nel grafo** (llama-graph.cpp build_moe_ffn, 4 siti: gate_up fuso,
  gate, up, down): `out += α · mul_mat_id(bs, blocksums(x), ids)` con
  blocksums = `sum_rows(reshape_4d(x, 32, …))`.
- **Registri/bug noti e fixati**: (1) lettura risultato col formato aggregato
  sbagliato → falsi falliti; (2) tensori step senza buffer → abort al warmup;
  (3) allineamento malloc < TENSOR_ALIGNMENT → posix_memalign(64).
- **Stato attuale**: compila; a runtime abortisce in `ggml_mul_mat_id` con
  `GGML_ASSERT(!ggml_is_transposed(as))` sul tensore degli step. Da debuggare.

## 2. Debug aperto: assert "transposed" (prossima azione)

Ipotesi corrente: disallineamento tra il layout del tensore step creato
(`ggml_new_tensor_3d(ctx, Q8_0, nb32, n_out, n_exp)`, nb32 = n_embd/32 BLOCKS)
e ciò che `ggml_mul_mat_id` si aspetta per `as` ([cols=K, rows=M, n_exp], K =
dim0 di `b`). Verificare con gdb in locale su un MoE piccolo Q4_K_M
(riproduzione: `LLAMA_MOE_BRACKET=1 llama-perplexity -m moe.q4km.gguf -f wiki.test.raw -ngl 0`):
- stampare ne/nb del tensore step al momento della chiamata
- confrontare con le asserzioni in ggml.c:3346 (`!ggml_is_transposed(as)`,
  `as->ne[0] == b->ne[0]`, `as->ne[3] == 1`)
- fix probabile: invertire le dim del tensore step (o trasporre) affinché
  `ne[0] = K` sia compatto e `nb[0] ≤ nb[1]`.

Il crash avviene durante il WARMUP (primo decode), prima della calibrazione:
riproduzione locale in ~2 min, gdb simboli completi (build locale ha percorsi
sorgente).

## 3. Dati e risultati già validi (in data-moe/, locali)

- `humaneval/` — A/B HumanEval completo: no-exp **89,63%** vs exp20 **90,85%**
  (decode 69,3 vs 56,0 t/s), JSON con risposte+reasoning integrali, log server.
- `run.jsonl` (72 MB) — log routing per-token/tutti-i-40-layer della sessione
  LCB su V100; `moe-report.txt`, `predictions.json` (predizioni set-conditioned
  ~47% a budget 24).
- `ab-*.log` — A/B prefetch temporale/chunked su V100 e 2080 Ti (+9%, −6..−14%).
- `risultati-moe.tgz`, `ab-2080ti.tgz` — pacchetti grezzi.

## 4. Verdetto sperimentali già chiusi

- Espansione 20/0,8/25-39: +qualità (GPQA +2,5 su Q8; HumanEval +1,2 su Q4),
  −19% decode su modello residente.
- Prefetch temporale/chunked su unified memory: ±0/⁻ (il driver LRU incassa già
  il segnale di routing); condizioni per riprovarlo in `moe-predict.md`.
- LCB-v6-Plus: i problemi 0-1 resistono a 7+ configurazioni di routing → limite
  di capacità a 4-bit, non di tuning (documento con 10+ combinazioni).

## 5. Per riprendere il bench LiveCodeBench-v6-Plus (91 problemi)

```bash
git clone https://github.com/vagrillo/AgrillaMoE && cd AgrillaMoE
git clone --depth 1 -b moe-expansion https://github.com/vagrillo/llama.cpp llama.cpp
bash setup-lcb.sh        # build + modello + dataset in parallelo
bash run-lcb-adaptive.sh # run adattivo (patch a 2 falliti, 10 combinazioni)
```
Dopo il fix del punto 2: la calibrazione alpha (script `calibrate-alpha.sh`,
grid 0→1 su perplexity wikitext-2, ~20 min) e poi il benchmark completo.

## 6. VM

- 2080 Ti (26502@58.8.185.95): i dati sono salvati in locale — **si può distruggere**.
- 3090 (56520@86.127.8.145): nulla di prezioso oltre la build — **si può distruggere**;
  al restart serve rifare solo `setup-lcb.sh` + pull del fork (i fix sono su GitHub).
