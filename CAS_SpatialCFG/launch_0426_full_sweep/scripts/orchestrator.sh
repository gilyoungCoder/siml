#!/bin/bash
# Orchestrator for full sweep on siml-05 (RTX 3090 24GB).
# Memory budget per worker:
#   single-concept (1 pack, 4 fam):    ~5 GB  → 2 workers/GPU OK (NSLOTS=16)
#   multi 1c (1 pack):                 ~5 GB  → would be 2/GPU OK
#   multi 2c (2 packs, 8 fam):         ~8 GB
#   multi 3c (3 packs, 12 fam):        ~11 GB
#   multi 7c (7 packs, 28 fam):        ~17 GB → 2/GPU = 34 GB → OOM on 24GB card
# To stay safe across all multi setups (especially 7c), Phase 2 uses NSLOTS=8 (1 worker/GPU).
# Phase 1 / 1B (single-concept) keep NSLOTS=16 (2 workers/GPU).
set -uo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR

START=$(date +%s)
echo "[$(date)] === Orchestrator START ===" | tee $LOGDIR/orchestrator.log

# === PHASE 1: Single-concept HYBRID reproducibility (Table 8 SD v1.4), NSLOTS=16 ===
echo "[$(date)] PHASE 1: 13 single-concept hybrid cells (NSLOTS=16, 2/GPU)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase1.sh $gpu $slot 16 $BASE/cells_phase1_single.tsv &
done
wait
P1_END=$(date +%s)
echo "[$(date)] PHASE 1 done (elapsed: $(((P1_END-START)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 2: Multi-concept HYBRID sweep, NSLOTS=8 (1/GPU) to fit 7c on 24GB ===
echo "[$(date)] PHASE 2: 39 multi cells (NSLOTS=8, 1/GPU due to 24GB OOM constraint)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase2.sh $gpu $slot 8 $BASE/cells_phase2_multi.tsv &
done
wait
P2_END=$(date +%s)
echo "[$(date)] PHASE 2 done (elapsed: $(((P2_END-P1_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 1B: Single-concept ANCHOR reproducibility, NSLOTS=16 ===
echo "[$(date)] PHASE 1B: 13 single-concept anchor cells (NSLOTS=16, 2/GPU)" | tee -a $LOGDIR/orchestrator.log
for slot in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  gpu=$((slot % 8))
  bash $BASE/scripts/dispatch_phase1_anchor.sh $gpu $slot 16 $BASE/cells_phase1_anchor.tsv &
done
wait
P1B_END=$(date +%s)
echo "[$(date)] PHASE 1B done (elapsed: $(((P1B_END-P2_END)/60))m)" | tee -a $LOGDIR/orchestrator.log

# === PHASE 3: v5 Qwen3-VL eval ===
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
