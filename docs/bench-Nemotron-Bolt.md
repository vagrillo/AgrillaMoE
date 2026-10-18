# Benchmark: NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB on AgrillaMoE

First GPQA-Diamond characterization of the **Nemotron 3.5 Bolt 74B-A9B-24GB**
(NVIDIA official GGUF: hybrid Mamba-2/attention, LatentMoE pool of **496
experts** shared across 44 MoE layers with **per-layer heterogeneous pools**
(~240 each), native **top-9** routing, SIGMOID gating, 1 shared expert;
IQ2-XXS experts + NVFP4 Mamba quant, 22 GB — NVIDIA's own 24GB recipe).

## Required fork changes (branch `moe-expansion`, commit `f4da1de8a`)

1. **Loader**: the per-layer expert pool is read from the real GGUF tensor dims
   (the `expert_count` KV = 496 is the pool *sum*, tensors carry ~240 each).
2. **Expansion guard**: native-k check per layer (`n_expert_used(il)`) instead
   of the global default (layer 0 is mamba, k=0); pool range check relaxed to
   `[used(il), n_expert]`.
3. **Context init**: MoE check on `n_expert_used_max()` (same mamba layer-0
   issue).

## Results — full GPQA-Diamond (198 questions) @ 8K reasoning budget

Setup: V100-SXM2-32GB, official GGUF (22 GB, full GPU), ctx 65536 (4 × 16384),
KV q8_0, temperature 0, evalscope, batch 4.

| config | accuracy | cap-8K saturation |
|---|---|---|
| stock top-9 (noexp) | **85.35%** | 114/198 (58%) |
| expansion 12, thr 0.80, layers 60-87 (exp01) | 84.85% | 105/198 (53%) |

**Expansion is neutral on this model** (−0.5 pts, within the ±2.5 pts
confidence band of a 198-question sample): unlike LFM2.5 (where the effect
flipped with quantization and budget), the LatentMoE budget-conditioned router
of Bolt — itself the product of an architecture search over routed-expert
counts — appears already saturated at its native top-9.

Median reasoning ~8.1-8.2K tokens for both configs; roughly half the questions
saturate the 8K budget in both arms.

## Raw data

[`data/bench-lfm25/nemo/`](../data/bench-lfm25/README.md) — evalscope
predictions, reviews and reports for both runs.

---
Date: 2026-10-18. Instance: vast.ai V100-SXM2-32GB.
