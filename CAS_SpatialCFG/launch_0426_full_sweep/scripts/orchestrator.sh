#!/bin/bash
# Orchestrator for full sweep: Phase 1 (single repro) → Phase 2 (multi sweep) → Phase 3 (v5 eval) → Phase 4 (report).
# Run with: nohup bash orchestrator.sh > orchestrator.log 2>&1 &
# Designed to run autonomously on siml-01 with 8 GPUs (16 worker slots, 2 per GPU).
set -uo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR

START=$(date +%s)
echo "[$(date)] === Orchestrator START ===" | tee $LOGDIR/orchestrator.log

# === PHASE 1: Single-concept reproducibility (Table 8 hybrid SD v1.4) ===
echo "[$(date)] PHASE 1: 13 single cells × paper Table 8 hybrid hyperparams" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot / 2))
  bash $BASE/scripts/dispatch_phase1.sh $gpu $slot 16 $BASE/cells_phase1_single.tsv &
done
wait
P1_END=$(date +%s)
echo "[$(date)] PHASE 1 done (elapsed: $(((P1_END-START)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 2: Multi-concept sweep ===
echo "[$(date)] PHASE 2: 39 multi cells (1c/2c/3c/7c × 3 configs × eval-concepts)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot / 2))
  bash $BASE/scripts/dispatch_phase2.sh $gpu $slot 16 $BASE/cells_phase2_multi.tsv &
done
wait
P2_END=$(date +%s)
echo "[$(date)] PHASE 2 done (elapsed: $(((P2_END-P1_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 3: v5 Qwen3-VL eval ===
echo "[$(date)] PHASE 3: v5 Qwen3-VL eval" | tee -a $LOGDIR/orchestrator.log
bash $BASE/scripts/eval_v5_dispatch.sh
P3_END=$(date +%s)
echo "[$(date)] PHASE 3 done (elapsed: $(((P3_END-P2_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 4: Report ===
echo "[$(date)] PHASE 4: Report generation" | tee -a $LOGDIR/orchestrator.log
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
$PY $BASE/scripts/report.py > $BASE/REPORT.md 2>&1 || echo "[FAIL report]"
P4_END=$(date +%s)
echo "[$(date)] PHASE 4 done (elapsed: $(((P4_END-P3_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

TOTAL=$((P4_END - START))
echo "[$(date)] === Orchestrator COMPLETE === total: $((TOTAL/3600))h $((TOTAL%3600/60))m" | tee -a $LOGDIR/orchestrator.log
echo "Report: $BASE/REPORT.md"
