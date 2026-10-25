#!/usr/bin/env python3
"""
Genera il piano di quantizzazione mixed-precision per Qwen3.6-35B-A3B (fork
moe-expansion) a partire dai conteggi d'uso puliti (patch v2, callback sincrona).

Fasce per layer (l'ordine e' quello dei conteggi raccolti):
  tier 1: rank   1..16   -> Q8_0   (8.5 bpw)
  tier 2: rank  17..64   -> Q6_K   (6.5625 bpw)
  tier 3: rank  65..192  -> Q4_K   (4.5 bpw)
  tier 4: rank 193..256  -> Q2_K   (2.5625 bpw)

Output: expert-rank/quantization-plan.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "expert-usage-total.json")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "quantization-plan.json")

TIERS = [
    (1, 16, "Q8_0", 8.5),
    (17, 64, "Q6_K", 6.5625),
    (65, 192, "Q4_K", 4.5),
    (193, 256, "Q2_K", 2.5625),
]

d = json.load(open(SRC))
total_tokens = d["total_tokens"]
layer_tokens = {int(k): int(v) for k, v in (d.get("layer_tokens") or {}).items()}

layers_out = {}
tot_sel = 0
for li in range(len(d["layers"])):
    v = d["layers"][str(li)]
    counts = v["counts"]
    ranked = sorted(enumerate(counts), key=lambda t: -t[1])  # (expert_id, count) desc
    experts = []
    tier_sizes = []
    for rank, (eid, cnt) in enumerate(ranked, start=1):
        tier = next(t for t in TIERS if t[0] <= rank <= t[1])
        experts.append({"rank": rank, "expert": eid, "count": cnt,
                        "tier": tier[2], "bpw": tier[3]})
        tier_sizes.append(tier[2])
    sels = sum(counts)
    tot_sel += sels
    layers_out[str(li)] = {
        "n_expert": len(counts),
        "tokens": layer_tokens.get(li, 0),
        "selections": sels,
        "sel_per_token": round(sels / max(1, layer_tokens.get(li, 1)), 2),
        "experts": experts,
        "tier_summary": {t: tier_sizes.count(t) for t in ("Q8_0", "Q6_K", "Q4_K", "Q2_K")},
    }

avg_bpw = sum(e["bpw"] for l in layers_out.values() for e in l["experts"]) / \
          sum(len(l["experts"]) for l in layers_out.values())

plan = {
    "format": "moe-expert-quantization-plan-v1",
    "source_usage": os.path.basename(SRC),
    "total_tokens": total_tokens,
    "total_selections": tot_sel,
    "n_layers": len(layers_out),
    "n_expert_per_layer": layers_out["0"]["n_expert"],
    "tiers": [{"range": f"{t[0]}-{t[1]}", "type": t[2], "bpw": t[3]} for t in TIERS],
    "avg_expert_bpw": round(avg_bpw, 3),
    "layers": layers_out,
}
json.dump(plan, open(DST, "w"), indent=1)
print(f"piano scritto: {DST}")

# file dell'ordine per llama-quantize --split-experts: una riga per layer,
# 256 id di esperti separati da virgola in ordine di rank
order_path = os.path.join(os.path.dirname(DST), "expert-order.txt")
with open(order_path, "w") as f:
    for li in range(len(layers_out)):
        ids = [e["expert"] for e in layers_out[str(li)]["experts"]]
        f.write(",".join(str(i) for i in ids) + "\n")
print(f"ordine scritto: {order_path}")
print(f"token {total_tokens} | selezioni {tot_sel} | bpw medio esperti {avg_bpw:.3f}")
# anteprima layer 0: top-8 e confini fascia
l0 = layers_out["0"]["experts"]
print("L0 top-8:", [(e["expert"], e["count"], e["tier"]) for e in l0[:8]])
print("L0 confini:", [(e["rank"], e["expert"], e["tier"]) for e in l0 if e["rank"] in (16, 17, 64, 65, 192, 193)])
