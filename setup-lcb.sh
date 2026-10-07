#!/bin/bash
# setup-lcb.sh — setup parallelo per il bench LiveCodeBench-v6-Plus (RTX 3090 24GB)
# Tutte le attività lunghe girano in parallelo: clone+build del fork, download del
# modello IQ4_XS (17.7GB), download del dataset + selftest del judge.
# Requisiti: git, curl, python3, cmake, gcc, CUDA toolkit 12+
set -uo pipefail
cd /root
JOBS="${LCB_JOBS:-$(nproc)}"
START=$(date +%s)
step_done() { echo "[$(( $(date +%s) - START ))s] $1"; }

# ---- 0) repo AgrillaMoE (veloce, foreground) ----
[ -d /root/AgrillaMoE ] || git clone -q https://github.com/vagrillo/AgrillaMoE /root/AgrillaMoE
cd /root/AgrillaMoE
git pull -q origin main 2>/dev/null || true
git clone -q --depth 1 -b moe-expansion https://github.com/vagrillo/llama.cpp llama.cpp 2>/dev/null || \
    (cd llama.cpp && git fetch -q --depth 1 origin moe-expansion && git reset -q --hard origin/moe-expansion)
step_done "repo pronti (fork: $(git -C llama.cpp log --oneline -1 | cut -c1-40))"

# ---- 1) PARALLELO: build fork+AgrillaMoE | download modello | dataset+selftest ----
( # A: build (ha bisogno del fork gia' clonato sopra)
  cd /root/AgrillaMoE
  AGRILLA_JOBS="$JOBS" ./build-linux.sh > /root/setup-build.log 2>&1 \
      && echo "[build] OK" || { echo "[build] FALLITA"; tail -5 /root/setup-build.log; }
  echo done-build > /root/.setup-A
) &
BUILD_PID=$!

( # B: modello IQ4_XS 17.7GB (entra in 24GB con KV q8_0 e margine)
  mkdir -p /root/AgrillaMoE/models
  curl -sL --fail -o /root/AgrillaMoE/models/IQ4XS.gguf \
      "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
      && echo "[modello] OK ($(du -h /root/AgrillaMoE/models/IQ4XS.gguf | cut -f1))" \
      || echo "[modello] FALLITO"
  echo done-model > /root/.setup-B
) &
MODEL_PID=$!

( # C: dataset LCB-Plus + selftest del judge (solo CPU)
  cd /root/AgrillaMoE
  for s in medium hard; do
      curl -sL --fail -o "lcb-plus-$s.jsonl" \
          "https://huggingface.co/datasets/BenchEvolver/livecodebench-plus/resolve/main/$s.jsonl"
  done
  LCB_JUDGE_SELFTEST=1 python3 lcb_run.py 0 /root/lcb-selftest.json \
      --dataset "/root/AgrillaMoE/lcb-plus-{split}.jsonl" --timeout 15 --limit 5 \
      > /root/setup-selftest.log 2>&1 \
      && echo "[judge] OK ($(grep 'judge selftest' /root/setup-selftest.log))" \
      || { echo "[judge] FALLITO"; tail -3 /root/setup-selftest.log; }
  echo done-judge > /root/.setup-C
) &
JUDGE_PID=$!

wait $BUILD_PID $MODEL_PID $JUDGE_PID

# ---- 2) verifica finale ----
FAIL=0
[ -f /root/.setup-A ] && [ -f /root/AgrillaMoE/dist/linux/agrillamoe ] || { echo "MANCA build"; FAIL=1; }
[ -f /root/.setup-B ] && [ -f /root/AgrillaMoE/models/IQ4XS.gguf ] || { echo "MANCA modello"; FAIL=1; }
[ -f /root/.setup-C ] || { echo "MANCA selftest"; FAIL=1; }
rm -f /root/.setup-A /root/.setup-B /root/.setup-C
echo
if [ "$FAIL" = "0" ]; then
    echo "SETUP COMPLETO in $(( $(date +%s) - START ))s — lancia: bash run-lcb-adaptive.sh"
else
    echo "SETUP INCOMPLETO — controlla i log sopra"
fi
exit $FAIL
