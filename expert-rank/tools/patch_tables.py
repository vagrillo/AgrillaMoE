import numpy as np
import gguf

MODEL = "/tmp/models/Qwen3.6-35B-A3B-mixed-expert.gguf"
SIZES = [16, 48, 128, 64]

order = [list(map(int, l.strip().split(","))) for l in open("/workspace/expert-order.txt") if l.strip()]
assert len(order) == 40 and all(len(r) == 256 for r in order), "expert-order inatteso"

reader = gguf.GGUFReader(MODEL)
targets = {}
for t in reader.tensors:
    if "ffn_exps_local_" in t.name or "ffn_exps_mask_" in t.name:
        targets[t.name] = (int(t.data_offset), bytes(np.asarray(t.data)))
assert len(targets) == 320, f"tabelle trovate: {len(targets)}"

# backup dei soli byte che verranno riscritti (320 KiB)
np.savez("/workspace/table-backup.npz",
         **{n: np.frombuffer(b, np.float32) for n, (o, b) in targets.items()})

f = open(MODEL, "r+b")
patched = 0
for L in range(40):
    off = 0
    for p in range(4):
        local = np.zeros(256, np.float32)
        mask = np.zeros(256, np.float32)
        for pos, e in enumerate(order[L][off:off + SIZES[p]]):
            local[e] = float(pos)
            mask[e] = 1.0
        for name, arr in ((f"blk.{L}.ffn_exps_local_{p}.weight", local),
                          (f"blk.{L}.ffn_exps_mask_{p}.weight", mask)):
            o, cur = targets[name]
            f.seek(o)
            if f.read(len(cur)) != cur:
                raise RuntimeError(f"offset mismatch per {name} @ {o}: abort, nessuna scrittura")
            f.seek(o)
            f.write(arr.tobytes())
            patched += 1
        off += SIZES[p]
f.close()
print(f"patched {patched}/320 tabelle")
