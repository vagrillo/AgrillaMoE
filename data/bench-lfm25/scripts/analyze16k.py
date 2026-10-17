#!/usr/bin/env python3
# Confronto noexp vs exp01: run 8K (50q) vs run 16K (198q), segmentato per
# domande che all'8K hanno raggiunto il tetto di reasoning
import json

def load(sub, root):
    preds = {}
    ppath = root + "/predictions/" + sub + "/gpqa_diamond_default.jsonl"
    for line in open(ppath):
        d = json.loads(line)
        pm = d.get("model_output", {}).get("perf_metrics") or {}
        blocks = (d["model_output"].get("choices") or [{}])[0].get("message", {}).get("content") or []
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        rchars = sum(len(b.get("reasoning") or "") for b in blocks if isinstance(b, dict))
        tchars = sum(len(b.get("text") or "") for b in blocks if isinstance(b, dict))
        comp = pm.get("output_tokens") or 0
        preds[d["index"]] = {"out": comp, "rtok": comp * (rchars / max(1, rchars + tchars))}
    recs = []
    rpath = root + "/reviews/" + sub + "/gpqa_diamond_default.jsonl"
    for line in open(rpath):
        d = json.loads(line)
        p = preds.get(d["index"], {"out": 0, "rtok": 0})
        sc = ((d.get("sample_score") or {}).get("score") or {}).get("value") or {}
        recs.append({"i": d["index"], "score": float(sc.get("accuracy", 0.0)), **p})
    recs.sort(key=lambda r: r["i"])
    return recs

def summ(tag, recs, budget):
    n = len(recs)
    if not n:
        print(tag + ": nessuna")
        return 0.0
    tot = sum(r["score"] for r in recs)
    hit = sum(1 for r in recs if r["rtok"] >= budget * 0.98)
    print(tag + ": acc " + format(tot / n * 100, ".1f") + "% (" + str(int(tot)) + "/" + str(n)
          + ") | cap-hit " + str(hit) + " | tok/risp ~" + format(sum(r["out"] for r in recs) / n, ".0f"))
    return tot / n

for cfg in ("noexp", "exp01"):
    print("== " + cfg + " ==")
    r8 = load("LFM25-" + cfg, "/root/gpqa-" + cfg)
    summ("  8K  (" + str(len(r8)) + "q)", r8, 8192)
    capped = [r["i"] for r in r8 if r["rtok"] >= 8192 * 0.98]
    try:
        r16 = load("LFM25-16k-" + cfg, "/root/gpqa16k-" + cfg)
        summ("  16K (" + str(len(r16)) + "q)", r16, 16384)
        sub = [r for r in r16 if r["i"] in capped]
        summ("  16K solo ex-cap8K (" + str(len(sub)) + "q)", sub, 16384)
        sub8 = [r for r in r8 if r["i"] in capped]
        summ("  (stesse domande @8K)", sub8, 8192)
    except Exception:
        import traceback; traceback.print_exc()
