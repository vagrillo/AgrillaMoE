#!/usr/bin/env python3
"""
Genera il GGUF mixed-precision per-esperto di Qwen3.6-35B-A3B dal sorgente Q8_0.

Per ogni layer, i 256 esperti sono ripartiti in 4 pool secondo il piano
(quantization-plan.json, fasce per rank d'uso):
  tier 1 (rank  1..16 ) -> tensor  _hi    Q8_0   (copia raw senza perdita)
  tier 2 (rank 17..64 ) -> tensor  _mid   Q6_K
  tier 3 (rank 65..192) -> tensor  _lo    Q4_K
  tier 4 (rank 193..256) -> tensor  _tail  Q2_K
Ogni pool include due tabelle F32 [256]: _local_p (indice locale dell'esperto,
0 se fuori pool) e _mask_p (1.0 se nel pool, else 0) usate dal grafo del fork
per il routing per-pool.

Gli altri tensori (attention, embedding, output, shared expert, router, norme,
nextn) vengono copiati verbatim dal sorgente (Q8_0/F32).

Uso: make_mixed_gguf.py <src_q8_0.gguf> <plan.json> <out.gguf>
"""
import json
import sys
import time

import numpy as np

import gguf
from gguf import quants

POOL_SIZES = [16, 48, 128, 64]          # hi, mid, lo, tail
POOL_TYPES = [gguf.GGMLQuantizationType.Q8_0,
              gguf.GGMLQuantizationType.Q6_K,
              gguf.GGMLQuantizationType.Q4_K,
              gguf.GGMLQuantizationType.Q2_K]
POOL_NAMES = ["hi", "mid", "lo", "tail"]


def expert_tensor(name):
    return name.endswith("ffn_gate_up_exps.weight") or name.endswith("ffn_down_exps.weight")


def main(src, plan_path, out_path):
    plan = json.load(open(plan_path))
    reader = gguf.GGUFReader(src)

    writer = gguf.GGUFWriter(out_path, arch=None)
    # copia tutti i metadati del sorgente
    for kv in reader.kv:
        writer.add_value(kv.key, kv.value_type, kv.value)
    writer.add_uint32("qwen35moe.expert_pools", len(POOL_SIZES))

    n_layers = plan["n_layers"]
    t_start = time.time()

    for t in reader.tensors:
        name = t.name
        shape = [int(x) for x in t.shape]
        if not expert_tensor(name):
            # copia verbatim (raw bytes + tipo sorgente)
            data = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
            writer.add_tensor(name, data, raw_dtype=t.tensor_type)
            print(f"copy   {name} {shape} {t.tensor_type}", flush=True)
            continue

        blk = name.split(".")[1]           # blk.N
        li = int(blk.replace("blk", "") or 0)
        layer_plan = plan["layers"][str(li)]["experts"]
        n_expert = len(layer_plan)
        assert shape[-1] == n_expert, (name, shape, n_expert)

        # dequantizza il tensore intero a F32 (layout memoria: esperto-major)
        n_blocks = t.n_elements // 32
        blocks = np.frombuffer(t.data.tobytes(), dtype=np.uint8).reshape(n_blocks, -1)
        f32 = gguf.dequantize(blocks, t.tensor_type).reshape(n_expert, -1)
        elems_per_expert = f32.shape[1]
        assert elems_per_expert % 256 == 0, elems_per_expert

        # tier order dal piano (rank crescente dentro ogni fascia)
        tier_ids = [[] for _ in POOL_SIZES]
        for e in layer_plan:
            ti = POOL_NAMES.index(e["tier"])
            tier_ids[ti].append(e["expert"])
        for ti, ids in enumerate(tier_ids):
            assert len(ids) == POOL_SIZES[ti], (name, ti, len(ids))

        for ti, ids in enumerate(tier_ids):
            sub = f32[np.asarray(ids, dtype=np.int64)]      # (pool, elems)
            qtype = POOL_TYPES[ti]
            cls = getattr(quants, qtype.name)
            qb = cls.quantize(sub)                           # hi: quantize Q8_0 == copia fedele dal dequant F32
            out_shape = [shape[0], shape[1], POOL_SIZES[ti]]
            byte_shape = gguf.quant_shape_from_byte_shape(out_shape, qtype)
            raw = np.frombuffer(qb.tobytes(), dtype=np.uint8).reshape(byte_shape)
            new_name = name.replace("ffn_gate_up_exps", f"ffn_gate_up_exps_{POOL_NAMES[ti]}") \
                           .replace("ffn_down_exps", f"ffn_down_exps_{POOL_NAMES[ti]}")
            writer.add_tensor(new_name, raw, raw_dtype=qtype)
            print(f"split  {new_name} {out_shape} {qtype} ({sub.nbytes/1e6:.0f} MB f32)", flush=True)

        # tabelle local/mask per pool (F32 [n_expert])
        for ti in range(4):
            local = np.zeros(n_expert, dtype=np.float32)
            mask = np.zeros(n_expert, dtype=np.float32)
            for pos, e in enumerate(tier_ids[ti]):
                local[e] = pos
                mask[e] = 1.0
            writer.add_tensor(f"{blk}.ffn_exps_local_{ti}.weight", local, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.add_tensor(f"{blk}.ffn_exps_mask_{ti}.weight", mask, raw_dtype=gguf.GGMLQuantizationType.F32)
        del f32

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"fatto in {time.time()-t_start:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
