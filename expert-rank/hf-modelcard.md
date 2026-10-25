---
license: apache-2.0
base_model: unsloth/Qwen3.6-35B-A3B-GGUF
library_name: gguf
tags:
- moe
- mixture-of-experts
- expert-expansion
- mixed-precision
- per-expert-quantization
- llama.cpp
- qwen3
- gpqa
datasets:
- idl-research/gpqa-diamond
language:
- en
---

# Qwen3.6-35B-A3B — MOExp MixedQuant (per-expert mixed precision)

GGUF of **Qwen3.6-35B-A3B** with **per-expert mixed precision**: the 256 routed
experts of every layer are partitioned into 4 pools by their **actual runtime
usage** (ranking collected with the
[`vagrillo/llama.cpp`](https://github.com/vagrillo/llama.cpp/tree/moe-expansion)
fork, branch `moe-expansion`, on a GPQA-Diamond workload): the most-used
experts at higher precision, the tail at lower precision. The **shared expert**
(always active, not routed) is kept at Q8_0.

| Resource | Size |
|---|---|
| This model (mixed per-expert) | **21.5 GB** |
| Uniform Q8_0 source | 37 GB |
| Routed-expert average bpw | 4.65 |

## Quantization scheme (per layer, 256 experts)

| Tier | Expert rank (by usage) | Type | bpw |
|---|---|---|---|
| 1 | 1–16 | Q8_0 | 8.5 |
| 2 | 17–64 | Q6_K | 6.5625 |
| 3 | 65–192 | Q4_K | 4.5 |
| 4 | 193–256 | Q2_K | 2.5625 |

Other tensors: attention Q8_0, token embeddings Q8_0, output Q8_0, shared
expert Q8_0, router F32. Requantized **from the Q8_0** with unsloth's imatrix
(`imatrix_unsloth.gguf_file`), sliced per expert.

## Tensor layout

Each layer's routed expert tensors are split into 4 pools with lookup tables:

```
ffn_gate_exps_hi/mid/lo/tail   [n_embd, n_ff, 16/48/128/64]   (or ffn_gate_up_exps_* when fused)
ffn_up_exps_hi/mid/lo/tail     [n_embd, n_ff, 16/48/128/64]
ffn_down_exps_hi/mid/lo/tail   [n_ff, n_embd, 16/48/128/64]
ffn_exps_local_0..3            [256] F32 — expert's local index within the pool
ffn_exps_mask_0..3             [256] F32 — 1.0 if the expert belongs to the pool
```

## Requirements

Requires the **[`vagrillo/llama.cpp`](https://github.com/vagrillo/llama.cpp/tree/moe-expansion)**
fork, branch `moe-expansion`, commit **`72c497b` or later** (per-expert pool
support in the loader/graph + the `mm_ids_helper` MMQ compaction fix).
Alternative engine: [AgrillaMoE](https://github.com/vagrillo/AgrillaMoE).

```bash
llama-server \
  -m Qwen3.6-35B-A3B-mixed-expert.gguf \
  -ngl 999 -c 73728 -ctk q8_0 -ctv q8_0 \
  --temp 0 --jinja --parallel 4 \
  --reasoning-budget 16384 \
  --reasoning-budget-message "Stop reasoning now: commit to your best-supported answer and respond with the final answer." \
  --moe-experts 20 --moe-expert-threshold 0.95 \
  --host 0.0.0.0 --port 9080
```

The recommended expansion profile (`--moe-experts 20 --moe-expert-threshold
0.95`, all layers) is the one used while collecting the ranking: the routing is
untouched by the mixed precision, so any expansion profile remains usable.

## Ranking methodology

1. The fork exposes the routing tensors (ids + post-expansion weights) and a
   per-tensor synchronized eval callback reads them at every ubatch
   (`LLAMA_MOE_EXPERT_USAGE_FILE`): cumulative count per (layer, expert) of all
   **kept** experts (weight > 1e-6, including the native top-8 and the
   expansion experts).
2. Workload: **GPQA-Diamond** (93 questions, greedy, reasoning budget 8192,
   concurrency 4) → 441,745 tokens, **20,591,738 selections**, 40/40 layers
   validated at ~11.5 selections/token.
3. The plan is the descending sort of the counts; no transformation.
4. Per-tier requant with the per-expert imatrix (sliced along the expert axis).

Full details, caveats (GPQA domain bias, nearly flat tail distribution) and
scripts: the [AgrillaMoE](https://github.com/vagrillo/AgrillaMoE) repository,
`expert-rank/` directory (`EXPERT-RANK.md`), and `docs/expert-pools.md` in the
fork. The raw per-(layer, expert) usage counts collected on GPQA-Diamond are in
[`expert-usage-total.json`](expert-usage-total.json) in this repository.

## Notes

- The per-pool compute evaluates `n_used` rows per pool (4×20 expert
  evaluations per token instead of 20): the decode FFN overhead is ~4×.
- Requires GPUs supported by CUDA 12.8+ (tested on V100-32GB, ~26 GB VRAM with
  73k ctx q8_0 and concurrency 4).
- GPQA-Diamond benchmark at 16K reasoning budget: running, results will be
  added to this model card.

## Acknowledgments

MoE expansion: llama.cpp port of DwarfStar's (ds4) "layer-scoped expert-budget
expansion" feature —
[V. Agrillo, Zenodo 2026](https://zenodo.org/records/22255483).
Quants: [unsloth](https://huggingface.co/unsloth) (source GGUF and imatrix).
Base model: Qwen/Qwen3.6-35B-A3B.
