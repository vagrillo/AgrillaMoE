#!/usr/bin/env python3
"""
moe_predict.py — analisi statistica e predizione degli esperti MoE da log JSONL.

Legge il log per-token delle attivazioni (esperti tenuti layer per layer,
inclusi quelli aggiunti dall'espansione) prodotto con:

    LLAMA_MOE_EXPERT_LOG=run.jsonl agrillamoe ...

(ogni riga: {"pos": [posizioni], "layer": {"25": [[id,...], ...], ...}} con
una lista di esperti per ogni token, per ogni layer espanso)

Calcola:
  1. frequenze per layer e copertura dei top-M (quanti esperti servono per
     coprire X% delle attivazioni: la dimensione della "finestra calda")
  2. matrice di transizione Markov layer L -> L+1 (co-attivazione di esperti
     in layer adiacenti) e la copertura della predizione top-M condizionata
  3. ripetitività per posizione (l'esperto usato al token t quanto prevede
     quello del token t-1 nello stesso layer: persistenza)

Output: report testuale + predictions.json (tabella per layer usabile da un
prefetcher: top_freq, transition scores, persistenza).

Uso:
    python3 moe_predict.py run.jsonl [--top M] [--out predictions.json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--top", type=int, default=16, help="M della predizione top-M")
    ap.add_argument("--out", default="predictions.json")
    args = ap.parse_args()

    freq = defaultdict(Counter)          # layer -> Counter(expert)
    trans = defaultdict(Counter)         # (layer, expert) -> Counter(expert nel layer+1)
    persist = defaultdict(Counter)       # layer -> Counter((prev_expert, expert)) appiattito
    prev_token_experts = {}              # layer -> lista esperti del token precedente
    n_tokens = 0

    for rec in load_records(args.jsonl):
        layers = rec.get("layer", {})
        for t in range(len(rec.get("pos", [])) or max((len(v) for v in layers.values()), default=0)):
            for layer_str, per_token in layers.items():
                if t >= len(per_token):
                    continue
                experts = per_token[t]
                L = int(layer_str)
                for e in experts:
                    freq[L][e] += 1
                # persistenza: stessi esperti del token precedente nello stesso layer
                if L in prev_token_experts:
                    for pe in prev_token_experts[L]:
                        for e in experts:
                            persist[L][(pe, e)] += 1
                prev_token_experts[L] = experts
            # transizioni layer -> layer successivo per lo stesso token
            lsorted = sorted(layers.keys(), key=int)
            for a, b in zip(lsorted, lsorted[1:]):
                la, per_a = int(a), layers[a]
                lb, per_b = b, layers[b]
                if t >= len(per_a) or t >= len(per_b):
                    continue
                for ea in per_a[t]:
                    for eb in per_b[t]:
                        trans[(la, ea)][eb] += 1
            n_tokens += 1

    if n_tokens == 0:
        print("nessun token nel log")
        sys.exit(1)

    print(f"token analizzati: {n_tokens}")
    predictions = {}

    print(f"\n{'layer':>6} | {'esperti distinti':>16} | top-M coverage (M={args.top})")
    print("-" * 60)
    for L in sorted(freq):
        c = freq[L]
        top = [e for e, _ in c.most_common(args.top)]
        covered = sum(c[e] for e in top)
        total = sum(c.values())
        print(f"{L:>6} | {len(c):>16} | {covered/total:6.1%}  top={top}")
        predictions[str(L)] = {"top_freq": top}

    # copertura della predizione markoviana: dato l'esperto piu' attivo del
    # layer L, quanto coprono i suoi top-M successori nel layer L+1?
    print(f"\npredizione markoviana L->L+1 (condizionata all'esperto top del layer):")
    for (L, ea), c in sorted(trans.items()):
        src_total = freq[L][ea]
        if src_total < n_tokens * 0.05:      # solo esperti frequenti
            continue
        top_next = [e for e, _ in c.most_common(args.top)]
        covered = sum(c[e] for e in top_next)
        print(f"  L{L} e{ea:>3} -> L{L+1}: {covered/sum(c.values()):6.1%} nei top-{args.top} {top_next[:8]}")
        key = str(L + 1)
        predictions.setdefault(key, {})["from_" + f"L{L}_e{ea}"] = top_next

    # persistenza: frazione di attivazioni che ripetono un esperto gia' usato
    # nel token precedente (quanto vale "prevedere il passato")
    print("\npersistenza per layer (ripetizione dal token precedente):")
    for L in sorted(freq):
        p = persist[L]
        total = sum(p.values())
        same = sum(v for (a, b), v in p.items() if a == b)
        if total:
            print(f"  L{L}: {same/total:6.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tokens": n_tokens, "top": args.top, "predictions": predictions},
                  f, indent=1)
    print(f"\nscritto {args.out}")


if __name__ == "__main__":
    main()
