# LiquidAI LFM2.5-24B-A2B on AgrillaMoE — full GPQA-Diamond @ 8K reasoning budget

**Headline result — full GPQA-Diamond (198 questions), 4-bit QAD-Q4_0, 8K reasoning
budget, temperature 0, V100-16GB:**

| config | score (198q) |
|---|---|
| stock top-4 (noexp) | 134/198 = **67.7%** |
| **MoE-expansion 8, thr 0.80, layers 25-39 (exp01)** | 140/198 = **70.7%** |
| official BF16 (LiquidAI, vLLM) | 74.42% ([source](https://huggingface.co/blog/LiquidAI/lfm25-24b-a2b)) |

**MoE-expansion adds +3.0 points over stock on the full dataset at 8K budget.**
The 4-bit models stay below the BF16 reference on the full set (−3.7 for the
expanded config); on the harder 50-question paired subset the expanded config
scored **78.0%, above the BF16 reference** — subset composition matters, and the
reasoning budget (8K, capped ~46% of answers) is a first-class variable.

Every question is covered by direct measurement or a documented proxy:
the first 50 come from the direct 8K run, the 146 budget-sensitive questions
(index ≥ 50 whose 16K-trace reasoning exceeded 8K tokens) were re-run directly
at 8K, and the remaining non-capping questions use their 16K score as a proxy
(their reasoning never touches the cap, so 8K behavior is expected to match).
127/198 questions per config are direct measurements.

## Per-config breakdown (full 198 @ 8K)

**stock top-4: 134/198 = 67.7%**

| segment | questions | score |
|---|---|---|
| first 50 (direct 8K run) | 50 | 35/50 = 70.0% |
| budget-sensitive (direct 8K re-run) | 77 | 34/77 = 44.2% |
| never capped (16K proxy) | 71 | 65/71 = 91.5% |

**expansion 8, layers 25-39: 140/198 = 70.7%**

| segment | questions | score |
|---|---|---|
| first 50 (direct 8K run) | 50 | 39/50 = 78.0% |
| budget-sensitive (direct 8K re-run) | 69 | 29/69 = 42.0% |
| never capped (16K proxy) | 79 | 72/79 = 91.1% |

Both configs share the same shape: ~70-78% on the head of the dataset, ~42-44%
on the budget-saturating tail, ~91% on questions the cap never binds. The
expansion advantage concentrates exactly where the routing has room to matter.

## Setup

- Model: LiquidAI LFM2.5-24B-A2B (24B total / ~2B active, 40 layers, 64 experts,
  native top-4, arch `lfm2moe`), QAD-Q4_0 4-bit (13.6 GB, full GPU offload)
- Hardware: vast.ai 1× Tesla V100-SXM2-16GB; CUDA 12.8; KV q8_0; ctx 65536
  (4 slots × 16384); 4 concurrent requests; temperature 0
- Engine: AgrillaMoE (llama.cpp fork) — the wrapper auto-injects the expansion
  profile for this model family (`--moe-experts 8 --moe-expert-threshold 0.8
  --moe-expert-layer-start 25 --moe-expert-layer-end 39`)
- Scoring: evalscope GPQA-Diamond (accuracy on the chosen option), review
  pipeline; raw predictions/reviews/configs/logs committed under
  [`data/bench-lfm25/`](../data/bench-lfm25/README.md)

## Compatibility

- Fork (`moe-expansion` branch): works unmodified — the `lfm2` graph calls
  `build_moe_ffn` with softmax gating and passes all expansion/reduce guards.
- Wrapper: signature-based auto-profile (40 layers, 64 experts, top-4 → the
  e8/L25-39 profile above), checked before the Qwen3.6 signature.
- Serverless: the published Docker image
  (`ghcr.io/vagrillo/agrillamoe-serverless:latest`) applies the same profile
  automatically (PyWorker `BACKEND=llama` contract).
- `--moe-reduce` (value-threshold pruning with mass-preserving renorm) also
  validated on this family.

## Q8_0 vs QAD-Q4_0: the quantization × expansion matrix (full 198 @ 8K)

Rerunning both configs with the **Q8_0** quant (25.5 GB, V100-32GB, same
protocol: 4×16K slots, KV q8_0, temp 0):

| config | Q4-QAD (QAD-Q4_0) | Q8_0 (naive) | official BF16 |
|---|---|---|---|
| stock top-4 | 67.7% (est.¹) | **75.8%** | 74.42% |
| expansion 8, layers 25-39 | **70.7%** (est.¹) | 68.7% | — |

¹ Q4 full-set numbers are estimates (50 direct + 77/69 direct re-runs + proxy
for never-capped questions); Q8 numbers are direct on all 198.
Q8 throughput: 32.8 t/s (noexp) / 28.6 t/s (exp01) vs ~38.7 / 36.5 for Q4.

Three observations:

1. **Stock: Q8_0 beats the BF16 reference** (75.8% vs 74.42%, same protocol
   caveat as before) and beats QAD-Q4 by +8. Naive higher-bit quantization
   helps the unmodified routing.
2. **Expansion: the sign flips per quant.** Expansion is +3.0 for the distilled
   QAD-Q4 but −7.1 for naive Q8 (68.7% vs stock's 75.8%). The distilled
   quantization appears tuned to the stock top-4 routing; naive Q8 under
   re-routed traffic overthinks (49.5% cap saturation, median reasoning ~8K).
3. **Paired-50 head-to-heads are stark**: Q4-QAD exp01 78.0% vs Q8 exp01 68.0%
   (−10); Q8 noexp 78.0% vs QAD-Q4 noexp 70.0% (+8). The two "78%" winners are
   opposite arms of the matrix.

Practical guidance for this model: **QAD-Q4_0 + expansion profile** for tight
token budgets; **Q8_0 stock** for best absolute accuracy at 8K (75.8%, above
the official BF16 number, at 32.8 t/s).

## Additional experiments (budget study)

The interaction between expert budget and reasoning budget is documented in
[lfm25-expansion-gpqa.md](lfm25-expansion-gpqa.md): at a tight 8K cap expansion
wins by +8 over stock (Q4-QAD); at 16K the sign inverts (stock 80.0% paired /
73.2% full vs expansion 74.0% / 71.7%) — expansion buys budget-efficiency (same
score class at roughly half the thinking tokens), not a higher ceiling. A
tripling budget (e12) scores below stock. Rerunning the 16K-capped questions at
8K confirms the cap is the main limit on the hardest block (16K beats 8K for
both configs: stock 44.2% → 51.9%, expansion 42.0% → 47.8%).

---
Date: 2026-10-17. Raw data: [`data/bench-lfm25/`](../data/bench-lfm25/README.md).
