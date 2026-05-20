#!/usr/bin/env bash
# Chain: wait for Dataset515 build → cleanup old datasets → v2 ablation pipeline.
# Triggered as a background nohup wrapper after the 515 build script is launched.

set -u

LOG_DIR=/media/fatih/1tb/training_data/logs
WAIT_PID=${1:?usage: run_after_515.sh <pid-of-build_dataset515>}

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] waiting for Dataset515 build (PID $WAIT_PID) to exit…"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[$(ts)] Dataset515 build PID $WAIT_PID exited."

# Sanity: verify Dataset515 looks complete
DS515=/media/fatih/1tb/training_data/nnunet_raw/Dataset515_CoronaryPseudoROI_v2
if [[ ! -f "$DS515/dataset.json" ]]; then
    echo "[$(ts)] ABORT: $DS515/dataset.json missing — build seems to have failed."
    exit 1
fi
N_IMG=$(find "$DS515/imagesTr" -name "*_0000.nii.gz" -type f 2>/dev/null | wc -l)
N_LBL=$(find "$DS515/labelsTr" -name "*.nii.gz" -type f 2>/dev/null | wc -l)
echo "[$(ts)] Dataset515 has $N_IMG ch0 files, $N_LBL labels."
if [[ "$N_IMG" -lt 1000 ]]; then
    echo "[$(ts)] ABORT: Dataset515 looks too small ($N_IMG cases)."
    exit 1
fi

echo "[$(ts)] === Step A: cleanup old datasets ==="
/media/fatih/1tb/training_data/cleanup_old_datasets.sh \
    2>&1 | tee "$LOG_DIR/cleanup_old_datasets_$(date +%Y%m%d_%H%M%S).log"

echo "[$(ts)] === Step B: v2 ablation pipeline ==="
/media/fatih/1tb/training_data/run_v2_ablation_pipeline.sh \
    2>&1 | tee "$LOG_DIR/v2_ablation_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "[$(ts)] ALL DONE."
