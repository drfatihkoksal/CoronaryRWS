#!/usr/bin/env bash
# Full v2 ablation pipeline for Dataset 521/522/523/525
#   1. Clean old preprocessed dirs
#   2. nnUNetv2_plan_and_preprocess (ResEncL, 2d)
#   3. nnUNetTrainer_50epochs, fold 0
#   4. Inference on Dataset510 arcade_test_* subset (sigma-matched)
#   5. compare_v2_ablation.py  (515 + 521/522/523/525)
set -ue

export nnUNet_raw=/media/fatih/1tb/training_data/nnunet_raw
export nnUNet_preprocessed=/media/fatih/1tb/training_data/nnunet_preprocessed
export nnUNet_results=/media/fatih/1tb/training_data/nnunet_results

NNU=/home/fatih/nnunet-env/bin
PY=/home/fatih/nnunet-env/bin/python3
WDIR=/media/fatih/1tb/training_data
LOG_DIR=$WDIR/logs
PRE=$nnUNet_preprocessed
RAW=$nnUNet_raw

# sigma-matched input folder for each dataset
declare -A SIGMA_DIR
SIGMA_DIR[521]=sigma20
SIGMA_DIR[522]=sigma30
SIGMA_DIR[523]=sigma60
SIGMA_DIR[525]=nogauss

DATASETS=(521 522 523 525)

ts()  { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

MAIN_LOG="${LOG_DIR}/full_pipeline_521_525_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

# ── STEP 1: clean broken preprocessed dirs ────────────────────────────────────
log "=== STEP 1: Remove old preprocessed dirs ==="
for ds in "${DATASETS[@]}"; do
    dir=$(ls -d "$PRE"/Dataset${ds}_* 2>/dev/null | head -1 || true)
    if [ -n "$dir" ]; then
        log "  rm -rf $dir"
        rm -rf "$dir"
    fi
done
log "Cleanup done."

# ── STEP 2: preprocess ────────────────────────────────────────────────────────
log "=== STEP 2: Preprocess ==="
for ds in "${DATASETS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    PLOG="${LOG_DIR}/Dataset${ds}_preprocess_ResEncL_${TS}.log"
    log "  Preprocessing Dataset${ds}..."
    "$NNU/nnUNetv2_plan_and_preprocess" -d "$ds" \
        -pl nnUNetPlannerResEncL -c 2d \
        --verify_dataset_integrity 2>&1 | tee "$PLOG"
    RAW_DIR=$(ls -d "$RAW"/Dataset${ds}_* 2>/dev/null | head -1 || true)
    PRE_DIR=$(ls -d "$PRE"/Dataset${ds}_* 2>/dev/null | head -1 || true)
    if [ -f "${RAW_DIR}/splits_final.json" ] && [ -n "$PRE_DIR" ]; then
        cp "${RAW_DIR}/splits_final.json" "${PRE_DIR}/splits_final.json"
        log "  splits_final.json copied for Dataset${ds}"
    fi
    log "  Dataset${ds} preprocess done."
done

# ── STEP 3: train ─────────────────────────────────────────────────────────────
log "=== STEP 3: Train ==="
for ds in "${DATASETS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    TLOG="${LOG_DIR}/Dataset${ds}_fold0_ResEncL_50ep_${TS}.log"
    log "  Training Dataset${ds}..."
    "$NNU/nnUNetv2_train" "$ds" 2d 0 \
        -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs \
        2>&1 | tee "$TLOG"
    log "  Dataset${ds} training done."
done

# ── STEP 4: inference ─────────────────────────────────────────────────────────
log "=== STEP 4: Inference ==="
PRED_BASE=$WDIR/predictions

for ds in "${DATASETS[@]}"; do
    SDIR="${SIGMA_DIR[$ds]}"
    IN_DIR="$PRED_BASE/Dataset510_test_inputs/$SDIR"
    OUT_DIR="$PRED_BASE/Dataset510_test_via${ds}_50ep"
    TS=$(date +%Y%m%d_%H%M%S)
    ILOG="${LOG_DIR}/Dataset${ds}_infer_${TS}.log"

    log "  Inference Dataset${ds} (input: $SDIR) → $OUT_DIR"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    "$NNU/nnUNetv2_predict" \
        -i "$IN_DIR" \
        -o "$OUT_DIR" \
        -d "$ds" -c 2d -f 0 \
        -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs \
        -chk checkpoint_best.pth \
        2>&1 | tee "$ILOG"
    log "  Dataset${ds} inference done."
done

# ── STEP 5: ablation comparison ───────────────────────────────────────────────
log "=== STEP 5: Ablation comparison (515 vs 521/522/523/525) ==="
cd "$WDIR"
"$PY" compare_v2_ablation.py 2>&1 | tee "${LOG_DIR}/ablation_compare_$(date +%Y%m%d_%H%M%S).log"

log "=== PIPELINE COMPLETE ==="
log "Results: $PRED_BASE/v2_ablation_comparison.json"
