# HumanEval A/B: MoE-expansion vs stock routing on Qwen3.6-35B-A3B

**First public HumanEval measurement we could find for Qwen3.6-35B-A3B** — and an
A/B of the MoE-expansion routing patch against stock top-8 routing, with
everything else held constant.

## TL;DR

| Config | HumanEval pass@1 | Decode speed |
|---|---|---|
| Stock routing (top-8) | **89.63%** (147/164) | 69.3 tok/s |
| **MoE-expansion** (20 experts, adaptive) | **90.85%** (149/164) | 56.0 tok/s |

Expansion is **+1.22 points** (2 more problems) at a **−19% decode speed** cost.
The difference is within statistical noise for n=164 (±3 pts CI) — treat it as
"equal or slightly better quality for ~20% slower generation".

## Setup

- **Model**: Qwen3.6-35B-A3B, Unsloth **UD-IQ4_XS** dynamic 4-bit (4.25 bpw, 16.5 GB) — fits entirely in VRAM
- **Hardware**: RTX 2080 Ti 22 GB (vast.ai), all layers on GPU (`-ngl 99 --fit off`), KV cache q8_0, ctx 16384
- **Server**: AgrillaMoE (llama.cpp fork `moe-expansion`), single slot, temp 0, greedy
- **Benchmark**: OpenAI **HumanEval** — all 164 problems, original tests (not EvalPlus+), pass@1, single greedy sample per problem
- **Thinking**: budget 4096 tokens, identical in both runs; answers extracted from markdown code fences, executed in a subprocess with the canonical `check(candidate)` tests, 12 s timeout
- **A/B isolation**: the *only* difference between the two runs is the routing:
  - `noexp`: `--no-moe-expansion` → stock top-8 on all 40 layers
  - `exp20`: expansion auto-injected → 20 routed experts, adaptive threshold 0.80 (5–20 kept per token), layers 25–39, linear decay 0.99→0.50

## Results

| Config | pass@1 | Passed | Gen tokens | Wall | Decode | Prompt |
|---|---|---|---|---|---|---|
| Stock top-8 | 89.63% | 147/164 | 499,588 | 7,324 s | 69.3 tok/s | 563 tok/s |
| Expansion 20 | 90.85% | 149/164 | 505,698 | 9,148 s | 56.0 tok/s | 505 tok/s |

### Paired per-problem analysis

- Solved by both: **139**
- Solved only by expansion: **10** (HumanEval/0, 4, 5, 11, 12, 17, 22, 62, 86, 93)
- Solved only by stock: **8** (HumanEval/1, 3, 9, 20, 56, 97, 145, 163)

Rolling pass-rate at matching progress points stayed expansion-ahead by
+2–3 problems from problem 20 onward (e.g. 60/60 vs 48/60 at problem 60).

## Context and caveats

- **No official HumanEval number exists for Qwen3.6-35B-A3B** (the model card
  reports agentic-coding benchmarks — Terminal-Bench, MCPMark — but not
  HumanEval). To our knowledge this is the first public HumanEval measurement
  for this model.
- These are **original HumanEval tests**, not HumanEval+/EvalPlus — do not
  compare 1:1 with the EvalPlus leaderboard (the + variant is stricter).
- **pass@1, single greedy sample**: leaderboard numbers sometimes average
  pass@1 over 20 samples at temp 0.8, or report base-model pass@100 — not
  comparable.
- Prompting: chat template with an instruction to output the completed function
  in a single Python code block; extraction takes the first fenced block.
- For reference points in the same ballpark: Qwen2.5-Coder-32B-Instruct
  (coder-specialized) reports 92.7% on HumanEval; generalist MoE models in the
  30B class typically land 80–88%.

## Reproduce

```bash
git clone https://github.com/vagrillo/AgrillaMoE && cd AgrillaMoE
git clone --depth 1 -b moe-expansion https://github.com/vagrillo/llama.cpp llama.cpp
# download Qwen3.6-35B-A3B-UD-IQ4_XS.gguf, then:
./dist/linux/agrillamoe -m model.gguf --no-moe-expansion --reasoning-budget 4096 \
    --flash-attn on -ctk q8_0 -ctv q8_0 --fit off -ngl 99 -c 16384 -np 1 \
    --host 127.0.0.1 --port 8097 --no-browser
python3 humaneval_run.py 8097 out.json 164 6144
# then rerun without --no-moe-expansion for the expansion arm
```

Raw per-problem results (including full completions and reasoning traces for
the expansion arm) are in the repository history of the author's local copy —
ask in the issues if you want the dataset.

*Eval date: 2026-10-06/07. Hardware and driver: vast.ai RTX 2080 Ti 22 GB,
CUDA 12.8, driver 580.178.04.*
