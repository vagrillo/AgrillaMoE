import sys
import numpy as np
import gguf

reader = gguf.GGUFReader("/tmp/models/Qwen3.6-35B-A3B-mixed-expert.gguf")
tabs = {}
for t in reader.tensors:
    for lay in ("blk.0.", "blk.1.", "blk.39."):
        if t.name.startswith(lay) and ("ffn_exps_local" in t.name or "ffn_exps_mask" in t.name):
            tabs[t.name] = np.asarray(t.data, dtype=np.float32)

order = [list(map(int, l.strip().split(","))) for l in open("/workspace/expert-order.txt") if l.strip()]
sizes = [16, 48, 128, 64]

def expected(layer, p):
    off = sum(sizes[:p])
    return {e: pos for pos, e in enumerate(order[layer][off:off + sizes[p]])}

def table_map(v):
    nz = np.nonzero(v)[0]
    return {int(e): int(v[e]) for e in nz}

def expected_nz(layer, p):
    # la posizione 0 vale 0.0 come gli out-of-pool: non osservabile nei nonzero
    off = sum(sizes[:p])
    return {e: pos for pos, e in enumerate(order[layer][off:off + sizes[p]]) if pos > 0}

for lay in (0, 1, 39):
    for p in range(4):
        name = f"blk.{lay}.ffn_exps_local_{p}.weight"
        cur = table_map(tabs[name])
        exp_self = expected_nz(lay, p)
        exp_next = expected_nz(min(lay + 1, len(order) - 1), p)
        msk = table_map(tabs[f"blk.{lay}.ffn_exps_mask_{p}.weight"])
        mask_self = msk == {e: 1.0 for e in expected_nz(lay, p)}
        print(f"L{lay} pool{p}: n={len(cur)} self={cur == exp_self} next={cur == exp_next} mask_self={mask_self}")

# puntuale: primi 5 elementi local_0 del layer 0 vs attesi
cur = table_map(tabs["blk.0.ffn_exps_local_0.weight"])
print("\nblk.0 local_0 atteso (ordine layer 0, primi 5):", dict(list(expected(0, 0).items())[:5]))
print("blk.0 local_0 atteso (ordine layer 1, primi 5):", dict(list(expected(1, 0).items())[:5]))
print("blk.0 local_0 reale (primi 5):", dict(list(cur.items())[:5]))
