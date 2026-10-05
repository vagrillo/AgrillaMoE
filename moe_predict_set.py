#!/usr/bin/env python3
"""
moe_predict_set.py — predizione set-conditioned del layer successivo.

Il prefetcher reale conoscera' l'INTERO insieme di esperti attivati al layer L
(e' noto prima che il layer L+1 computi) e dovra' prefetchare un insieme di
candidati per L+1. Questo script misura quanto coprirebbe:

  - train/test split 80/20 sulle righe del log
  - modello: per ogni (layer L, esperto e) i successori piu' frequenti in L+1
  - predizione per un token: unione dei top-k successori di ogni e nell'insieme
    di L, limitata a BUDGET esperti distinti (default 24)
  - metrica: copertura = |pred ∩ attivo(L+1)| / |attivo(L+1)|, e dimensione
    media dell'insieme predetto

Uso: python3 moe_predict_set.py run.jsonl [--budget 24] [--topk 4]
"""
import argparse, json
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--budget", type=int, default=24, help="max esperti predetti per token")
    ap.add_argument("--topk", type=int, default=4, help="successori per esperto sorgente")
    args = ap.parse_args()

    rows = []
    for line in open(args.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    # righe -> sequenze (pos0, layer -> lista per token)
    rows.sort(key=lambda r: r.get("pos0", 0))
    cut = int(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]

    # addestramento: successore count per (L, e)
    succ = defaultdict(Counter)
    def feed(recs):
        for rec in recs:
            layers = rec.get("layer", {})
            ls = sorted(layers.keys(), key=int)
            for la, lb in zip(ls, ls[1:]):
                A, B = layers[la], layers[lb]
                nt = min(len(A), len(B), rec.get("n", 0))
                for t in range(nt):
                    for e in A[t]:
                        succ[(int(la), e)].update(B[t])
    feed(train)

    # top-k successori per esperto sorgente
    top_succ = {k: [e for e, _ in c.most_common(args.topk)] for k, c in succ.items()}

    # valutazione sul test
    per_layer_cov = defaultdict(list)
    per_layer_sz = defaultdict(list)
    for rec in test:
        layers = rec.get("layer", {})
        ls = sorted(layers.keys(), key=int)
        for la, lb in zip(ls, ls[1:]):
            A, B = layers[la], layers[lb]
            nt = min(len(A), len(B), rec.get("n", 0))
            L = int(la)
            for t in range(nt):
                actual = set(B[t])
                if not actual:
                    continue
                pred = []
                seen = set()
                # ordina i sorgenti per addebitare il budget ai piu' affidabili prima
                for e in A[t]:
                    for s in top_succ.get((L, e), []):
                        if s not in seen:
                            seen.add(s)
                            pred.append(s)
                pred = set(pred[:args.budget])
                per_layer_cov[L].append(len(pred & actual) / len(actual))
                per_layer_sz[L].append(len(pred))

    print(f"righe: {len(rows)} (train {len(train)} / test {len(test)}), budget={args.budget}, topk={args.topk}")
    print(f"{'layer':>6} | {'copertura':>9} | {'dim pred':>8}")
    print("-" * 32)
    tot = []
    for L in sorted(per_layer_cov):
        cov = per_layer_cov[L]
        sz = per_layer_sz[L]
        tot.extend(cov)
        print(f"{L:>6} | {100*sum(cov)/len(cov):8.1f}% | {sum(sz)/len(sz):8.1f}")
    if tot:
        print("-" * 32)
        print(f"media complessiva: {100*sum(tot)/len(tot):.1f}% di copertura con budget {args.budget}")


if __name__ == "__main__":
    main()
