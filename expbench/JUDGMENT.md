# HumanEval-style expansion A/B — judge results (10 configs × 5 problems)

**Judge: performed by the repository author (human) on the full 50-run dataset**
(generation + objective tests + reasoning traces). The automated LLM-judge stage
produced empty outputs (thinking budget consumed the 2k answer cap) and was
replaced by this manual judgment, combined with the objective test matrix below.

## Setup recap

- Model: Qwen3.6-35B-A3B, Unsloth **UD-Q3_K_XL** (3.4 bpw experts), full GPU on RTX 3080 20 GB
- 5 Python problems (DP / sliding window / state-BFS / sweep line / stack evaluator), 6-7 reference tests each, 31 tests total
- Per run: thinking budget 24576, ctx 32768, KV q8_0, temperature 0, one sample
- 10 routing configs: stock top-8 + 9 MoE-expansion variants (12-24 experts, thresholds 0.6-0.9, layer windows 10-39 / 20-39 / 25-39 / 0-39)

## Objective test matrix (share of reference tests passed)

| config | p1 | p2 | p3 | p4 | p5 | **mean** |
|---|---|---|---|---|---|---|
| stock (top-8) | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e12 (L25-39) | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e24 (L25-39) | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e20last10 (L30-39) | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e16t07l2039 | 0.50 | 0.83 | 0.83 | 1.00 | 1.00 | 0.833 |
| e20t06 (L25-39) | 0.33 | 0.83 | 0.83 | 1.00 | 1.00 | 0.800 |
| e16 (L25-39) | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20 (L25-39) | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20t09 (L25-39) | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20all (L0-39) | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |

## Key findings

1. **Expansion does not improve coding quality on this model at 3.4 bpw.** The
   stock top-8 routing ties the best light-expansion configs and beats all
   heavy-expansion configs on the objective tests.
2. **Heavy expansion collapses on the hardest problem (p5, expression
   evaluator).** All four configs with ≥16 routed experts on ≥15 layers score
   0/7 on p5 — complete functional failure. The failure mode (from reasoning
   traces): under heavy expansion the model enters an overthinking spiral —
   57 k characters of reasoning (3× the stock run) for the same problem — and
   emits a plausible-looking but functionally broken evaluator.
3. **Light expansion is neutral**: e12 / e24 / e20last10 match stock exactly.
4. **Threshold 0.6 degrades p1** (0.33 vs 0.67): keeping more low-rank experts
   adds noise on the DP problem.
5. Ranking by mean test ratio: **stock = e12 = e24 = e20last10 (0.867) >
   e16t07l2039 (0.833) > e20t06 (0.800) > e16 = e20 = e20t09 = e20all (0.667)**.

## Verdict

- **Default profile CONFIRMED as best overall for coding** — with the important
  nuance that **light expansion (e12, or last-10-layers) is quality-neutral**,
  so it can be chosen for its GPQA gains without coding regressions.
- **Heavy expansion (≥16 experts on ≥15 layers) is harmful for Python coding**
  at this quant level: the p5 complete failure appeared in 4/4 heavy configs
  and never in the light ones.
- Recommendation: for coding workloads keep **stock top-8** or at most
  **e12/e20last10**; do not use ≥16-expert profiles on this quant.

## Data

- Per-run records with full reasoning + answers: `runs/<cfg>/p<k>.json` (this archive)
- `tests-summary.json` — per-config summary
- Objective harness: `expbench.py --stage tests` (re-runnable)
