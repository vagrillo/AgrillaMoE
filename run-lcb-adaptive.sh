#!/bin/bash
# run-lcb-adaptive.sh v2 — LiveCodeBench-v6-Plus (91 problemi) su RTX 3090 24GB
#
# v2 (richiesta utente): coda combinazioni riprioritizzata —
#   1. 20 esperti T=0.9 layer 20-39
#   2. 20 esperti T=0.9 layer 10-39
#   poi le altre non ancora provate. Problemi 0-1 (falliti con i default) sono
#   patchati PRIMA del run principale; se una combinazione vince, il run
#   principale prosegue CON QUELLA (validazione su larga scala).
set -uo pipefail
cd /root/AgrillaMoE
M="${LCB_MODEL:-models/IQ4XS.gguf}"
PORT=8099
BUDGET="${LCB_THINKING_BUDGET:-8192}"
NP="${LCB_PROBLEMS:-91}"
MAIN_START="${LCB_MAIN_START:-2}"     # 0 e 1 gia' falliti con i default
START=$(date +%s)

BASE_ARGS=(--flash-attn on -ctk q8_0 -ctv q8_0 --fit off -ngl 99 -c 16384 -np 1 --no-browser \
           --reasoning-budget "$BUDGET")
DEFAULT_MOE=(--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39)

COMBOS=(
  "--moe-experts 20 --moe-expert-threshold 0.9 --moe-expert-layer-start 20 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.9 --moe-expert-layer-start 10 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.6 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.9 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 0  --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 20 --moe-expert-layer-end 39"
  "--moe-experts 16 --moe-expert-threshold 0.7 --moe-expert-layer-start 20 --moe-expert-layer-end 39"
  "--no-moe-expansion"
)

start_server() {
  pkill -x agrillamoe 2>/dev/null; sleep 2
  ./dist/linux/agrillamoe -m "$M" "$@" "${BASE_ARGS[@]}" \
      --host 127.0.0.1 --port $PORT > /root/srv-lcb.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 300); do
    sleep 3
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 $SRV 2>/dev/null || return 1
  done
  return 1
}

run_problem() {  # $1 = indice, $2 = output
  LCB_THINKING_BUDGET="$BUDGET" python3 lcb_run.py $PORT "$2" --index "$1" --timeout 15 \
      --max-tokens "$((BUDGET + 4096))" \
      --dataset /root/AgrillaMoE/lcbdata >/dev/null 2>&1
  python3 -c "
import json
try:
    d = json.load(open('$2'))
    print(int(d['results'][0]['ok']))
except Exception:
    pass"
}

echo "###### LCB-v6-Plus ADAPTIVE v2 — start $(date) ######"

# ---- PATCH INIZIALE: problemi 0 e 1 (falliti con i default 20/0.8/25-39) ----
WINNER_MOE=()
declare -A FIXED_BY
INIT_OK=0
echo "===== PATCH INIZIALE problemi 0-1: ${#COMBOS[@]} combinazioni (user-primo: L20-39/L10-39 @T0.9) ====="
pkill -x agrillamoe 2>/dev/null; sleep 2
for ci in "${!COMBOS[@]}"; do
  COMBO_N=$((ci+1)); COMBO="${COMBOS[$ci]}"
  echo "--- combinazione $COMBO_N/${#COMBOS[@]}: $COMBO"
  start_server $COMBO || { echo "  server non partito, skip"; continue; }
  rA=$(run_problem 0 /tmp/lcb-p0.json)
  rB=$(run_problem 1 /tmp/lcb-p1.json)
  echo "  problema 0: $rA | problema 1: $rB"
  if [ "$rA" = "1" ] && [ "$rB" = "1" ]; then
    WINNER_MOE=($COMBO)
    FIXED_BY[0]="combo $COMBO_N"; FIXED_BY[1]="combo $COMBO_N"
    PATCH_REPORT+=("problemi 0,1 -> risolti da combo $COMBO_N: $COMBO")
    INIT_OK=1
    echo "  >>> ENTRAMBI RISOLTI: il run principale prosegue con questa combinazione"
    break
  fi
  PATCH_REPORT+=("problemi 0,1 -> combinazione $COMBO_N non risolutiva")
done
if [ "$INIT_OK" = "0" ]; then
  echo "nessuna combinazione risolve 0-1: sono limite di capacita', non di tuning"
  PATCH_REPORT+=("problemi 0,1 -> nessuna combinazione risolutiva (limite capacita')")
fi

# ---- RUN PRINCIPALE (con la combinazione vincente se esiste, altrimenti default) ----
RUN_MOE=("${WINNER_MOE[@]:-${DEFAULT_MOE[@]}}")
echo ""
echo "===== RUN PRINCIPALE problemi $MAIN_START-$((NP-1)) — moe: ${RUN_MOE[*]:-default} ====="
start_server "${RUN_MOE[@]}" || { echo "server non partito"; exit 1; }

declare -A RESULT
FAILED=()
PATCH_REPORT2=()
TOTAL_OK=$((TOTAL_OK + INIT_OK * 2))

for ((IDX=MAIN_START; IDX<NP; IDX++)); do
  if run_problem "$IDX" "/tmp/lcb-one.json"; then
    RESULT[$IDX]="ok"; TOTAL_OK=$((TOTAL_OK+1))
  else
    RESULT[$IDX]="failed"
    FAILED+=("$IDX")
    echo "[problema $IDX] FALLITO (in attesa: ${#FAILED[@]}/2)"
  fi

  if [ "${#FAILED[@]}" -eq 2 ]; then
    A="${FAILED[0]}"; B="${FAILED[1]}"
    echo ""; echo "===== PATCH: problemi $A e $B ====="
    pkill -x agrillamoe 2>/dev/null; sleep 2
    local_win=""
    for ci in "${!COMBOS[@]}"; do
      CN=$((ci+1)); COMBO="${COMBOS[$ci]}"
      [[ "${RUN_MOE[*]}" == "$COMBO" ]] && { echo "  skip (gia' in uso)"; continue; }
      echo "--- combinazione $CN: $COMBO"
      start_server $COMBO || continue
      rA=$(run_problem "$A" /tmp/lcb-pa.json)
      rB=$(run_problem "$B" /tmp/lcb-pb.json)
      echo "  $A:$rA  $B:$rB"
      if [ "$rA" = "1" ] && [ "$rB" = "1" ]; then
        local_win="$CN"; FIXED_BY[$A]="combo $CN"; FIXED_BY[$B]="combo $CN"
        PATCH_REPORT2+=("$A,$B -> risolti da combo $CN: $COMBO")
        echo "  >>> RISOLTI"
        break
      fi
      PATCH_REPORT2+=("$A,$B -> combo $CN non risolutiva")
    done
    [ -z "$local_win" ] && { PATCH_REPORT2+=("$A,$B -> nessuna combinazione risolutiva"); echo "  nessuna risolutiva"; }
    FAILED=()
    echo "===== ripresa run principale ====="
    start_server "${RUN_MOE[@]}" || exit 1
  fi
done

pkill -x agrillamoe 2>/dev/null

# ---- report ----
NPASS=0; NFAIL=0; VARIED=0
SUMMARY_ROWS=""
for ((IDX=0; IDX<NP; IDX++)); do
  R="?"; [[ -n "${RESULT[$IDX]:-}" ]] && R="${RESULT[$IDX]}"
  [[ -n "${FIXED_BY[$IDX]:-}" ]] && R="$R [${FIXED_BY[$IDX]}]"
  case "$R" in ok*) NPASS=$((NPASS+1));; failed*) NFAIL=$((NFAIL+1));; esac
  [[ "$R" == *"combo"* ]] && VARIED=$((VARIED+1))
  SUMMARY_ROWS+="{\"index\":$IDX,\"status\":\"$R\"},"
done
python3 - "$NPASS" "$NFAIL" "$VARIED" "$SUMMARY_ROWS" "$START" "${WINNER_MOE[*]:-none}" <<'PYEOF'
import json, sys, time, os
npass, nfail, varied = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = "[" + sys.argv[4].rstrip(",") + "]"
start = int(sys.argv[5]); winner = sys.argv[6]
report = {
    "benchmark": "LiveCodeBench-v6-Plus (91 problems, BenchEvolver)",
    "model": "Qwen3.6-35B-A3B UD-IQ4_XS, full GPU (RTX 3090 24GB)",
    "thinking_budget": int(os.environ.get("LCB_THINKING_BUDGET", "8192")),
    "default_expansion": {"experts": 20, "threshold": 0.8, "layers": "25-39"},
    "winner_expansion": winner if winner != "none" else "default confirmed",
    "passed": npass, "failed": nfail,
    "pass_at_1": round(100.0 * npass / max(1, npass + nfail), 2),
    "problems_fixed_with_varied_params": varied,
    "verdict": ("default params CONFIRMED" if varied == 0 and winner == "none" else
                (f"params VARIED: main run continued with [{winner}] — validated on full benchmark" if winner != "none" else
                 "mixed: some problems need varied params")),
    "wall_seconds": round(time.time() - start, 1),
    "details": json.loads(rows),
}
open("/root/lcb-adaptive-results.json", "w").write(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("passed", "failed", "pass_at_1", "winner_expansion", "verdict")}, indent=1))
PYEOF
echo "--- patch report ---"
printf '%s\n' "${PATCH_REPORT[@]}" "${PATCH_REPORT2[@]}"
echo "###### FINE — $(date) — durata $(( ($(date +%s) - START) / 60 )) min ######"
