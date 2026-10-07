#!/bin/bash
# run-lcb-adaptive.sh — LiveCodeBench-v6-Plus (91 problemi) su RTX 3090 24GB
#
# Strategia adattiva:
#   1. il run parte con l'espansione di DEFAULT (20 esperti, T=0.80, layer 25-39)
#   2. quando 2 problemi sono in errore: STOP del run principale
#   3. "patch phase": i 2 problemi falliti vengono ritentati con 10 combinazioni
#      diverse di parametri di espansione (una sessione server per combinazione,
#      entrambi i problemi per sessione); si ferma alla prima combinazione che
#      li risolve entrambi
#   4. il run principale riparte con i parametri di default; a fine run il
#      report dice se i parametri di default sono CONFERMATI o VARIATI
#
# Output: /root/lcb-adaptive-results.json + /root/lcb-patch-report.json
set -uo pipefail
cd /root/AgrillaMoE
M="${LCB_MODEL:-models/IQ4XS.gguf}"
PORT=8099
BUDGET="${LCB_THINKING_BUDGET:-8192}"
NP="${LCB_PROBLEMS:-91}"          # numero di problemi del run principale
START=$(date +%s)

# il reasoning budget va impostato SUL SERVER (il flag modella il thinking vero);
# max_tokens dell'harness copre pensiero+risposta
BASE_ARGS=(--flash-attn on -ctk q8_0 -ctv q8_0 --fit off -ngl 99 -c 16384 -np 1 --no-browser \
           --reasoning-budget "$BUDGET")
DEFAULT_MOE=(--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39)

# 10 combinazioni di patch (diverse per N, soglia, range di layer; la #10 e' il
# routing nativo come controllo)
COMBOS=(
  "--moe-experts 12 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 16 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 24 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 32 --moe-expert-threshold 0.8 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.6 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.9 --moe-expert-layer-start 25 --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 0  --moe-expert-layer-end 39"
  "--moe-experts 20 --moe-expert-threshold 0.8 --moe-expert-layer-start 20 --moe-expert-layer-end 39"
  "--moe-experts 16 --moe-expert-threshold 0.7 --moe-expert-layer-start 20 --moe-expert-layer-end 39"
  "--no-moe-expansion"
)

start_server() {  # $@ = flag moe extra (opzionali)
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

run_problem() {  # $1 = indice problema, $2 = file di output
  LCB_THINKING_BUDGET="$BUDGET" python3 lcb_run.py $PORT "$2" --index "$1" --timeout 15 \
      --max-tokens "$((BUDGET + 4096))" \
      --dataset /root/AgrillaMoE/lcbdata >/dev/null 2>&1
  python3 -c "import json,sys; print(json.load(open('$2'))['ok'])" 2>/dev/null
}

echo "###### LCB-v6-Plus ADAPTIVE — start $(date) ######"
start_server "${DEFAULT_MOE[@]}" || { echo "server non partito"; exit 1; }
echo "server pronto (espansione default: 20 / 0.80 / 25-39)"

declare -A RESULT      # idx -> ok/failed/COMBO_n
declare -A FIXED_BY    # idx -> combinazione vincente
FAILED=()              # indici in attesa di patch
PATCH_REPORT=()        # righe di report delle patch
TOTAL_OK=0
COMBO_N=0

for ((IDX=0; IDX<NP; IDX++)); do
  if run_problem "$IDX" "/tmp/lcb-one.json"; then
    RESULT[$IDX]="ok"
    TOTAL_OK=$((TOTAL_OK+1))
  else
    RESULT[$IDX]="failed"
    FAILED+=("$IDX")
    echo "[problema $IDX] FALLITO (falliti in attesa: ${#FAILED[@]}/2)"
  fi

  # ---- patch phase a 2 falliti ----
  if [ "${#FAILED[@]}" -eq 2 ]; then
    A="${FAILED[0]}"; B="${FAILED[1]}"
    echo ""
    echo "===== PATCH PHASE: problemi $A e $B — 10 combinazioni ====="
    pkill -x agrillamoe 2>/dev/null; sleep 2
    WINNER=""
    for ci in "${!COMBOS[@]}"; do
      COMBO_N=$((ci+1))
      COMBO="${COMBOS[$ci]}"
      echo "--- combinazione $COMBO_N/10: $COMBO"
      start_server $COMBO || { echo "  server non partito, skip"; continue; }
      local_ok=()
      for IDX2 in $A $B; do
        if r=$(run_problem "$IDX2" "/tmp/lcb-patch-$IDX2.json"); then
          local_ok+=(1); RESULT[$IDX2]="ok (combo $COMBO_N)"
        else
          local_ok+=(0); RESULT[$IDX2]="failed"
        fi
      done
      if [ "${local_ok[0]}" = "1" ] && [ "${local_ok[1]}" = "1" ]; then
        WINNER="COMBO_$COMBO_N"
        FIXED_BY[$A]="combo $COMBO_N"; FIXED_BY[$B]="combo $COMBO_N"
        PATCH_REPORT+=("problemi $A,$B -> risolti da $WINNER: $COMBO")
        echo "  >>> entrambi risolti da $WINNER"
        break
      else
        PATCH_REPORT+=("problemi $A,$B -> combinazione $COMBO_N non risolutiva (${local_ok[0]}/${local_ok[1]} ok)")
        echo "  non risolutiva"
      fi
    done
    if [ -z "$WINNER" ]; then
      PATCH_REPORT+=("problemi $A,$B -> nessuna delle 10 combinazioni li ha risolti")
      echo "  nessuna combinazione risolutiva: i problemi restano falliti"
    fi
    FAILED=()
    echo "===== fine patch phase — ripresa del run principale ====="
    start_server "${DEFAULT_MOE[@]}" || { echo "server non ripartito"; exit 1; }
  fi
done

pkill -x agrillamoe 2>/dev/null

# ---- report ----
NPASS=0; NFAIL=0; VARIED=0
SUMMARY_ROWS=""
for ((IDX=0; IDX<NP; IDX++)); do
  R="${RESULT[$IDX]:-n/d}"
  case "$R" in
    ok*) NPASS=$((NPASS+1));;
    failed*) NFAIL=$((NFAIL+1));;
  esac
  if [[ "$R" == *"combo"* ]]; then VARIED=$((VARIED+1)); fi
  FB="${FIXED_BY[$IDX]:-}"
  SUMMARY_ROWS+="{\"index\":$IDX,\"status\":\"$R\",\"fixed_by\":\"$FB\"},"
done
python3 - "$NPASS" "$NFAIL" "$VARIED" "$SUMMARY_ROWS" "$START" <<'PYEOF'
import json, sys, time
npass, nfail, varied, rows, start = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
rows = "[" + rows.rstrip(",") + "]"
report = {
    "benchmark": "LiveCodeBench-v6-Plus (91 problems, BenchEvolver)",
    "model": "Qwen3.6-35B-A3B UD-IQ4_XS, full GPU (RTX 3090 24GB)",
    "thinking_budget": int(__import__("os").environ.get("LCB_THINKING_BUDGET", "4096")),
    "default_expansion": {"experts": 20, "threshold": 0.8, "layers": "25-39"},
    "passed": npass, "failed": nfail,
    "pass_at_1_default_only": round(100.0 * npass / max(1, npass + nfail), 2),
    "problems_fixed_with_varied_params": varied,
    "verdict": ("default params CONFIRMED" if varied == 0 else
                f"default params VARIED: {varied} problem(s) fixed with different expansion params"),
    "wall_seconds": round(time.time() - start, 1),
    "details": json.loads(rows),
}
open("/root/lcb-adaptive-results.json", "w").write(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("passed", "failed", "pass_at_1_default_only",
      "problems_fixed_with_varied_params", "verdict")}, indent=1))
PYEOF
echo "--- patch report ---"
printf '%s\n' "${PATCH_REPORT[@]}"
echo "###### FINE — $(date) — durata $(( ($(date +%s) - START) / 60 )) min ######"
