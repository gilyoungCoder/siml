#!/bin/bash
# Launch SDXL-Lightning 4-step generation on siml-01 g4-g7 (4 GPUs).
# Datasets: I2P full, MJA (4 concepts), UnlearnDiff, Ring-A-Bell.
# Total ~5330 prompts / 4 GPUs = ~1333 per GPU. Each chunk runs datasets sequentially.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0425_sdxl_lightning/sdxl_lightning_chunk.sh
LOGDIR=$REPO/logs/launch_0425_sdxl_lightning
ssh siml-01 "mkdir -p $LOGDIR"

# Per-dataset GPU assignment (4 GPUs total: g4,g5,g6,g7)
# - I2P: 4710 prompts, split across all 4 GPUs (~1178 each)
# - MJA: each concept 100 prompts, one concept per GPU spare
# - UnlearnDiff: 142 prompts, 1 GPU
# - RingABell: 79 prompts, 1 GPU

# Strategy: dispatch sequentially so each GPU handles its share
#   g4: I2P 0-1178 + mja_sexual full + unlearndiff full
#   g5: I2P 1178-2355 + mja_violent full + ringabell full
#   g6: I2P 2355-3532 + mja_illegal full
#   g7: I2P 3532-4710 + mja_disturbing full

cat > /tmp/sdxl_light_g4.sh <<EOF
#!/bin/bash
set -uo pipefail
bash $SCRIPT 4 i2p 0 1178
bash $SCRIPT 4 mja_sexual 0 100
bash $SCRIPT 4 unlearndiff 0 142
EOF
cat > /tmp/sdxl_light_g5.sh <<EOF
#!/bin/bash
set -uo pipefail
bash $SCRIPT 5 i2p 1178 2355
bash $SCRIPT 5 mja_violent 0 100
bash $SCRIPT 5 ringabell 0 79
EOF
cat > /tmp/sdxl_light_g6.sh <<EOF
#!/bin/bash
set -uo pipefail
bash $SCRIPT 6 i2p 2355 3532
bash $SCRIPT 6 mja_illegal 0 100
EOF
cat > /tmp/sdxl_light_g7.sh <<EOF
#!/bin/bash
set -uo pipefail
bash $SCRIPT 7 i2p 3532 4710
bash $SCRIPT 7 mja_disturbing 0 100
EOF

rsync -az /tmp/sdxl_light_g4.sh /tmp/sdxl_light_g5.sh /tmp/sdxl_light_g6.sh /tmp/sdxl_light_g7.sh siml-01:/tmp/
for GPU in 4 5 6 7; do
  ssh siml-01 "chmod +x /tmp/sdxl_light_g${GPU}.sh && nohup bash /tmp/sdxl_light_g${GPU}.sh </dev/null >$LOGDIR/master_g${GPU}.out 2>&1 & disown"
  echo "Launched siml-01 g$GPU sequential"
done
echo "[$(date)] All 4 SDXL-Lightning sequential workers dispatched on siml-01 g4-g7"
