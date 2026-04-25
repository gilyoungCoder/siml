#!/bin/bash
# Orchestrator for full sweep:
#   Phase 1  — single-concept hybrid reproducibility (Table 8)
#   Phase 2  — multi-concept hybrid sweep (1c/2c/3c/7c × 3 configs)
#   Phase 1B — single-concept anchor reproducibility (master sources)
#   Phase 3  — v5 Qwen3-VL eval over Phase 1 + Phase 2 + Phase 1B outputs
#   Phase 4  — final REPORT.md
# Run with: nohup bash orchestrator.sh > orchestrator.log 2>&1 &
# Slot→GPU mapping: gpu = slot % 8 (round-robin across all 8 GPUs even when N_cells < N_slots).
set -uo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR

START=$(date +%s)
echo "[$(date)] === Orchestrator START ===" | tee $LOGDIR/orchestrator.log

# === PHASE 1: Single-concept HYBRID reproducibility (Table 8 SD v1.4) ===
echo "[$(date)] PHASE 1: 13 single-concept hybrid cells" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase1.sh $gpu $slot 16 $BASE/cells_phase1_single.tsv &
done
wait
P1_END=$(date +%s)
echo "[$(date)] PHASE 1 done (elapsed: $(((P1_END-START)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 2: Multi-concept HYBRID sweep ===
echo "[$(date)] PHASE 2: 39 multi cells (1c/2c/3c/7c × 3 configs × eval-concepts)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase2.sh $gpu $slot 16 $BASE/cells_phase2_multi.tsv &
done
wait
P2_END=$(date +%s)
echo "[$(date)] PHASE 2 done (elapsed: $(((P2_END-P1_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 1B: Single-concept ANCHOR reproducibility ===
echo "[$(date)] PHASE 1B: 13 single-concept anchor cells (paper / master configs)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase1_anchor.sh $gpu $slot 16 $BASE/cells_phase1_anchor.tsv &
done
wait
P1B_END=$(date +%s)
echo "[$(date)] PHASE 1B done (elapsed: $(((P1B_END-P2_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 3: v5 Qwen3-VL eval on all phases ===
echo "[$(date)] PHASE 3: v5 Qwen3-VL eval" | tee -a $LOGDIR/orchestrator.log
bash $BASE/scripts/eval_v5_dispatch.sh
P3_END=$(date +%s)
echo "[$(date)] PHASE 3 done (elapsed: $(((P3_END-P1B_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 4: Report ===
echo "[$(date)] PHASE 4: Report generation" | tee -a $LOGDIR/orchestrator.log
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
$PY $BASE/scripts/report.py > $BASE/REPORT.md 2>&1 || echo "[FAIL report]"
P4_END=$(date +%s)
echo "[$(date)] PHASE 4 done (elapsed: $(((P4_END-P3_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

TOTAL=$((P4_END - START))
echo "[$(date)] === Orchestrator COMPLETE === total: $((TOTAL/3600))h $((TOTAL%3600/60))m" | tee -a $LOGDIR/orchestrator.log
echo "Report: $BASE/REPORT.md"
