# Expert-budget expansion on LiquidAI LFM2.5-24B-A2B: budget-efficiency, not max-score

**TL;DR** — We tested MoE *expert-budget expansion* (a llama.cpp graph-level
technique that raises the routed-expert budget beyond the model's native top-k
on selected layers) on **LFM2.5-24B-A2B quantized to 4-bit (QAD-Q4_0)** across
two reasoning budgets:

- **At a tight 8K reasoning cap**, expansion (8 experts, layers 25-39) scores
  **78.0%** on GPQA-Diamond vs 70.0% stock (+8 pts) — **above the 74.42%
  officially reported by LiquidAI for the BF16 model** — at a ~6% throughput cost.
- **At a generous 16K cap the sign inverts**: stock reaches **80.0%** on the
  same 50 paired questions (73.2% on the full 198) while expansion drops to
  74.0% (71.7% full) — the expanded routing overthinks when given room.

The headline is therefore **budget-efficiency**: expansion buys you the
full-precision-class score at roughly half the reasoning budget, not a higher
ceiling. All artifacts (raw predictions, reviews, scripts) are open:
[`data/bench-lfm25/`](../data/bench-lfm25/README.md).

## What is expert-budget expansion

Standard MoE inference routes every token to the top-k experts the model was
trained with. Expert-budget expansion changes that at the graph level:

1. widen the per-token selection from top-k to top-N (N > k) on a chosen layer
   range,
2. keep an expanded expert only if its router score ≥ threshold × (reference
   rank score) — an adaptive cut,
3. apply linear influence decay to the added ranks (they contribute, but less),
4. renormalize, so the FFN output scale is preserved.

It is the opposite of expert pruning: you spend extra compute on late layers to
give the model a wider expert pool exactly where it helps.

## Experimental setup

| | |
|---|---|
| Model | LiquidAI LFM2.5-24B-A2B (24B total / ~2B active, 40 layers, 64 experts, native top-4, arch `lfm2moe`) |
| Quantization | **QAD-Q4_0** 4-bit (quantization-aware distillation, by LiquidAI), 13.6 GB |
| Hardware | vast.ai, 1× Tesla V100-SXM2-16GB (full GPU offload) |
| Engine | AgrillaMoE (llama.cpp fork) + evalscope, OpenAI-compatible API |
| Benchmark | GPQA-Diamond, temperature 0, reasoning budget capped (8K or 16K), KV q8_0 |
| Protocol | paired questions (identical set across configs), 4 concurrent requests |

## Results @ 8K reasoning budget (50 paired questions)

| config | accuracy | throughput | tpot |
|---|---|---|---|
| stock top-4 | 70.0% | 38.7 tok/s | 25.2 ms |
| **expansion 8, thr 0.80, layers 25-39** | **78.0%** | 36.5 tok/s | 27.0 ms |
| expansion 12, thr 0.80, layers 20-39 | 64.0% | 39.2 tok/s | 25.2 ms |

Reference: the official BF16 number is **74.42%**
([LiquidAI model blog](https://huggingface.co/blog/LiquidAI/lfm25-24b-a2b),
vLLM, recommended generation params, no reasoning budget). In the same blog
comparison: Qwen3.6-35B-A3B 83.69, Gemma-4-26B-A4B 74.90.

**The conservative expansion (doubling the native top-4 on the late layers) in
4-bit scores above the official BF16 number.** Note the honest caveat: protocols
differ (vLLM vs llama.cpp, full set vs 50-question subset, no cap vs 8K cap), so
this is an indicative comparison, not an apples-to-apples claim.

## Results @ 16K reasoning budget (full 198 questions + paired 50)

| config | paired-50 @8K | paired-50 @16K | full-198 @16K | tok/answer @16K |
|---|---|---|---|---|
| stock top-4 | 70.0% | **80.0%** (+10: 6 fixed, 1 lost) | **73.2%** | ~9.7K |
| expansion 8, layers 25-39 | 78.0% | 74.0% (−4: 4 fixed, 6 lost) | 71.7% | ~9.2K |

The 16K runs contain the 8K benchmark's 50 questions as a prefix (identical
indices), enabling a strictly paired comparison.

Segmented view (stock, paired on the 23 questions that saturated the 8K cap):

| subset | @8K | @16K |
|---|---|---|
| the 23 budget-capped questions | 34.8% | **56.5%** (+21.7 pts) |

For expansion the same segment moves the *other* way: its 21 capped questions
scored 66.7% at 8K and drop to 52.4% at 16K (~14.9K tokens of reasoning each —
the expanded routing keeps thinking and lands worse).

## Findings

1. **Expansion and reasoning budget interact non-trivially — the sign flips**:
   at a tight 8K cap expansion wins by +8; at a generous 16K cap stock wins by
   +1.5 (full set) / +6 (paired). The expanded routing appears to converge
   faster under pressure but to overthink when given room: its ex-cap answers
   nearly double their reasoning length and get worse.
2. **Practical takeaway — expansion buys budget-efficiency, not a higher
   ceiling**: 78.0% @8K (≈6.9K tok/answer) vs the 80.0% that stock only reaches
   at 16K (≈9.7K tok/answer). If you pay per generated token or need latency,
   expansion gets you the same class of score with ~30% fewer thinking tokens.
3. **Non-monotonic response to expert budget**: doubling (4→8) helps at tight
   budget, tripling (4→12 on a wider range) hurts (−6 pts). A 2B-active model
   doesn't want that much spread.
4. **Quantization cost is recoverable by routing**: stock 4-bit loses ~4.4 pts
   vs BF16 at 8K; with an adequate reasoning budget it exceeds the BF16
   reference outright (80.0% paired / 73.2% full vs 74.42%).
5. **Reasoning budget is a first-class variable**: always report the cap
   alongside the score — here the ranking of two configs inverts with it.

## Limitations

- One benchmark (GPQA-Diamond), small paired subsets (50 questions at 8K).
- The BF16 comparison crosses harnesses (vLLM vs llama.cpp) and budgets — treat
  it as directional evidence, and the full-198 16K runs above as the controlled
  part of this study.
- One seed per config (temperature 0, so effectively deterministic given the
  serving stack).

## Next: the full-precision ask

To close the loop we want to run **LFM2.5-24B-A2B in BF16 (or FP8) with the
same 16K reasoning budget and the same expansion profile** — the missing arm of
the experiment (official numbers come from a different harness and budget).
BF16 needs ~49 GB for weights alone: ideally a few hours on an **80GB GPU
(A100/H100)** — or 2×48GB with a pipeline split. If you can donate access to a
machine (or spot capacity), open an issue on the repo or ping the author — you
get a named spot in the results table.

## Reproduce

- Fork + server: <https://github.com/vagrillo/AgrillaMoE>
  (auto-detects the model family and injects the profile: for LFM2.5 it is
  exactly `--moe-experts 8 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39`)
- Raw data + scripts: [`data/bench-lfm25/`](../data/bench-lfm25/README.md)
- Serverless Docker image: `ghcr.io/vagrillo/agrillamoe-serverless:latest`
