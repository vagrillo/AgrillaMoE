# Raw benchmark data — LiquidAI LFM2.5-24B-A2B (GPQA-Diamond, vast.ai V100-16GB)

Raw evalscope artifacts kept for future analysis. Model: QAD-Q4_0 4-bit,
temperature 0, batch 4, KV q8_0, `--reasoning-budget` as noted.

## Layout

| directory | run | budget | questions | score |
|---|---|---|---|---|
| `gpqa-noexp/`    | stock top-4        | 8K  | 50  | 70.0% |
| `gpqa-exp01/`    | e8, thr 0.80, L25-39 | 8K  | 50  | 78.0% |
| `gpqa-exp02/`    | e12, thr 0.80, L20-39 | 8K  | 50  | 64.0% |
| `gpqa16k-noexp/` | stock top-4        | 16K | 198 | 73.2% |
| `gpqa16k-exp01/` | e8, thr 0.80, L25-39 | 16K | 198 | (run in progress) |

Each run directory contains:

- `predictions/` — raw model outputs (JSONL, one record per question:
  reasoning + final answer blocks, `perf_metrics` with latency/ttft/tpot)
- `reviews/` — per-question `sample_score` after evalscope review
  (`sample_score.score.value.accuracy` = 1.0/0.0, joined by `index`)
- `reports/` — aggregated metrics JSON (`gpqa_diamond.json`) + `report.html`
- `logs/`, `configs/` — evalscope run logs and config snapshot

## Analysis

`scripts/analyze16k.py` (run on the host that has the directories under
`/root`, adjust the paths as needed) reproduces the headline numbers and the
paired "ex-cap-8K" segment analysis (questions that saturated the 8K budget,
re-scored at 16K):

```
noexp: 8K 70.0% (23/50 cap-hit) -> 16K 73.2% (198q)
  same 23 questions: 34.8% @8K -> 56.5% @16K
exp01: 8K 78.0% (21/50 cap-hit) -> 16K pending
```

`scripts/lfm-gpqa.sh` and `scripts/lfm-gpqa16k.sh` are the exact server+eval
loops used on the V100 instance.
