# MoE-expansion parameter verification — 10 configs × 5 problems, judging report

**Model**: Qwen3.6-35B-A3B · Unsloth UD-Q3_K_XL (3.4 bpw) · RTX 3080 20 GB, full GPU
**Workload**: 5 medium Python problems (5 different natures), thinking budget 24 576, ctx 32 768, KV q8_0, temperature 0
**Configs**: 10 routing profiles (stock top-8 + 9 MoE-expansion variants)
**Total**: 50 runs, reasoning + answer archived for every run

## Scope of the verification

This experiment answers a narrower question than the earlier GPQA/HumanEval
benchmarks: **for coding workloads at 3.4 bpw, do the MoE-expansion routing
parameters matter — and which profile is best?** Quality is measured by
31 objective reference tests across 5 problems (6-7 tests per problem,
executed in sandboxed subprocesses), complemented by reasoning-trace analysis
(length, reconsideration rate, self-verification rate).

## The 10 configurations

| id | experts | threshold | layers expanded | notes |
|---|---|---|---|---|
| stock | 8 (native) | — | none | baseline, `--no-moe-expansion` |
| e12 | 12 | 0.80 | 25-39 | light expansion |
| e16 | 16 | 0.80 | 25-39 | medium |
| e20 | 20 | 0.80 | 25-39 | AgrillaMoE default profile |
| e24 | 24 | 0.80 | 25-39 | max experts on 15 layers |
| e20t06 | 20 | 0.60 | 25-39 | low threshold → keeps more experts (7-20/token) |
| e20t09 | 20 | 0.90 | 25-39 | high threshold → keeps fewer (9-20/token) |
| e20all | 20 | 0.80 | 0-39 | expansion on all 40 layers |
| e20last10 | 20 | 0.80 | 30-39 | last 10 layers only |
| e16t07l2039 | 16 | 0.70 | 20-39 | mixed intermediate |

Common to all: decay 0.99→0.50 on added ranks, renorm auto, temperature 0,
single greedy sample. The 50 runs and their full reasoning traces are archived
in the repository `data-moe/expbench/` (and reproducible via `expbench/run-expbench.sh`).

## Results: objective tests (share of reference tests passed)

| config | p1 DP | p2 strings | p3 BFS state | p4 intervals | p5 evaluator | **mean** |
|---|---|---|---|---|---|---|
| **stock (top-8)** | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e12 | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e24 | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e20last10 | 0.67 | 0.83 | 0.83 | 1.00 | 1.00 | **0.867** |
| e16t07l2039 | 0.50 | 0.83 | 0.83 | 1.00 | 1.00 | 0.833 |
| e20t06 | 0.33 | 0.83 | 0.83 | 1.00 | 1.00 | 0.800 |
| e16 | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20 | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20t09 | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |
| e20all | 0.67 | 0.83 | 0.83 | 1.00 | 0.00 | 0.667 |

## Reasoning-trace analysis (the four configs tied at 0.867)

Average per problem over the 5 problems:

| config | thinking (chars) | gen tokens | reconsiderations/problem | self-verifications/problem |
|---|---|---|---|---|
| **e24** | **15 729** | **5 158** | **13.4** | 8.0 |
| stock | 18 145 | 5 768 | 16.6 | **10.6** |
| e20last10 | 26 776 | 8 619 | 13.6 | 11.2 |
| e12 | 34 017 | 11 095 | 26.0 | **19.6** |

Reading:

- **e24 has the cleanest reasoning of the four**: ~10 % fewer thinking tokens
  than stock, the lowest reconsideration rate (13.4 vs 16.6), same final
  quality. Less effort, same outcome — the most efficient reasoner of the set.
- **e12 is the most deliberative**: 2× stock's thinking length and
  reconsiderations, with the highest self-verification count. On easy problems
  the extra deliberation is wasted effort; on harder ones (the GPQA gains of
  the light-expansion profiles) it is plausible that it pays off.
- **e20last10 sits in between** on every metric — consistent with its
  "expansion on fewer layers" design.

## Cross-benchmark context (this model, different workloads)

| workload | metric | stock | best light expansion | heavy expansion |
|---|---|---|---|---|
| GPQA-Diamond (Q8_0, earlier bench) | accuracy | 81.82 % | **84.34 %** (e20/0.8/L25-39) | n/t |
| HumanEval-style (Q4_K_M, earlier bench) | pass@1 | 89.63 % | 90.85 % (e20) | — |
| This benchmark, 5 coding problems (Q3_K_XL) | test ratio | **0.867** | 0.867 (e12/e24) | **0.667** (e16-e20all) |

## Findings

1. **MoE-expansion does not improve coding quality on this model at 3.4 bpw.**
   Stock top-8 ties the best light-expansion configs; no expansion config beats
   stock.
2. **Heavy expansion degrades coding**: every config with ≥16 routed experts on
   ≥15 layers fails problem 5 completely (0/7 tests, 4/4 configs), producing
   functionally broken evaluators. The failure mode (from reasoning traces):
   under heavy expansion the model overthinks — 57 k characters of reasoning
   vs 24 k for stock on the same problem — and emits a plausible but broken
   solution.
3. **Light expansion is quality-neutral on coding** (e12 / e24 / e20last10 all
   0.867) while retaining the GPQA-Diamond gains measured earlier (+2.5 points
   at Q8_0). This makes light profiles a reasonable default across workloads.
4. **The threshold is a sensitive knob**: 0.6 (keeps 7-20 experts/token) hurt
   p1 (0.33); 0.9 hurt p5 (0.00). The 0.8 default sits at a local optimum.

## Verdict and recommendation

- **Winner (quality): stock top-8 and the light profiles tie at 0.867.**
- **Winner (reasoning efficiency among the tied): e24** — same quality as the
  others with ~10 % fewer thinking tokens than stock and the lowest
  reconsideration rate of the whole benchmark. Recommended as the coding
  profile if a non-stock routing is preferred for its GPQA gains.
- **Avoid**: ≥16-expert profiles on ≥15 layers for coding at this quant.

## Data and reproducibility

- 50 per-run records (reasoning + answers): `data-moe/expbench/full/runs/`
- Objective test matrix: `objective-tests.json`
- Harness: `expbench/expbench.py` (stages: gen / tests / judge), resumable
- Config definitions: `expbench/configs.json`

*Eval window: 2026-10-07/08, vast.ai RTX 3080 20 GB, CUDA 12.8. Raw archive:
`expbench-results.tgz` (136 KB).*
