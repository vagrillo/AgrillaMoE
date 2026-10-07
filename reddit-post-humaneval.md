# Reddit post draft (r/LocalLLaMA) — HumanEval A/B: MoE-expansion on Qwen3.6-35B-A3B

---

**Title:**

> First public HumanEval number for Qwen3.6-35B-A3B — 90.9% at 4-bit on a RTX 2080 Ti, and an A/B of my "MoE expansion" routing patch vs stock

**Body:**

Hey r/LocalLLaMA,

TL;DR: I measured **Qwen3.6-35B-A3B at 4-bit (UD-IQ4_XS) hitting 89.6% pass@1
on HumanEval** (original tests, greedy, thinking budget 4096) on a single RTX
2080 Ti 22GB — and then ran a controlled A/B of a routing technique I've been
playing with: **MoE expansion**, which activates 20 experts per token instead
of the stock 8 on the last 15 layers. Result: **90.9%** (+2 problems) at −19%
decode speed. To my knowledge there is **no official HumanEval number
published for this model**, so this may be the first public measurement.

**Setup (both runs identical except routing):**
- Unsloth UD-IQ4_XS dynamic 4-bit (4.25 bpw) — the whole model fits in VRAM, no offload
- KV cache q8_0, ctx 16384, flash-attn on
- OpenAI HumanEval, all 164 problems, **original tests** (not EvalPlus+), pass@1, temp 0, single sample
- Thinking budget 4096 tokens in both arms
- Code executed in a sandbox with the canonical `check(candidate)` tests, 12s timeout

**Results:**

| Config | pass@1 | decode |
|---|---|---|
| Stock routing (top-8) | 89.63% (147/164) | 69 tok/s |
| MoE expansion (20 experts, adaptive, layers 25–39) | 90.85% (149/164) | 56 tok/s |

Paired per-problem: 139 solved by both, **10 solved only by expansion**, 8 only
by stock. Rolling pass rate stayed expansion-ahead by +2–3 problems at every
checkpoint.

**What is MoE expansion?** No retraining, no file changes — at inference time
the router keeps more experts per token than the model's native top-K (here:
20 instead of 8, with an adaptive threshold so easy tokens keep fewer), on a
slice of layers (25–39 of 40). You're consulting more of the network per
token. Same trick that gave **84.34% vs 81.82% on GPQA-Diamond at Q8** in
earlier benchmarks — now confirmed in coding too, at 4-bit.

**Honest caveats:**
- +2 problems on 164 is within statistical noise (±3 pts CI). Read it as
  "equal or slightly better quality", not a proven gain
- It's original HumanEval tests, **not** HumanEval+/EvalPlus — don't compare
  1:1 with the EvalPlus leaderboard
- pass@1 greedy n=1 — not the 20-sample protocol some leaderboards use
- No official HumanEval exists for this model (Qwen reports agentic-coding
  benchmarks instead), so no official reference to check against
- Expansion costs ~19% decode speed on a fully-resident model (more experts =
  more FLOPs per token)

**The tool** — I wrapped all of this into **AgrillaMoE**, a dedicated
llama.cpp server for this model: it detects your VRAM and suggests/downloads
the right Unsloth quant, applies the expansion profile by default (overridable),
exposes OpenAI *and* Anthropic-compatible APIs (Claude Code works out of the
box), and runs on NVIDIA from GTX 10xx to RTX 50xx, AMD via Vulkan, and Apple
Silicon via Metal. Static binaries for Linux and Windows on the releases page.

Links: [AgrillaMoE](https://github.com/vagrillo/AgrillaMoE) ·
[full eval writeup](https://github.com/vagrillo/AgrillaMoE/blob/main/humaneval-eval.md) ·
[16GB GPU beginner guide](https://github.com/vagrillo/AgrillaMoE/blob/main/gpu16gbguide.md)

Happy to share the raw per-problem results and reasoning traces. What would
you like to see next — LiveCodeBench, bigger thinking budgets, or expansion on
a coder-tuned model?

---

*Note per me (non per il post): il claim "first public HumanEval number" è
verificato con ricerche web il 2026-10-06; se pubblica un numero ufficiale
prima del post, togliere la frase e citare il confronto. Il post menziona
anche il contesto GPQA (84.34 vs 81.82) — verificato in RUN1209.*
