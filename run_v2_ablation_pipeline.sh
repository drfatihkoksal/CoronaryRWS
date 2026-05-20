#!/usr/bin/env bash
# v2 sigma ablation pipeline (Dataset515-based).
#
# Steps:
#   1. Build σ ablation datasets 521/522/523/524 + no-gauss 525 from 515
#   2. preprocess all 6 (515, 521-525) with ResEnc-L
#   3. train 50ep ResEnc-L fold 0 on each (sequential — single GPU)
#   4. Build ARCADE-test-only inference inputs (predictions/Dataset510_test_inputs/{sigma20,30,40,60,80,nogauss})
#   5. predict each model on its σ-matched inputs
#   6. comparison report
#
# Run AFTER Dataset515 build is complete.

set -u
set -o pipefail

export nnUNet_raw=/media/fatih/1tb/training_data/nnunet_raw
export nnUNet_preprocessed=/media/fatih/1tb/training_data/nnunet_preprocessed
export nnUNet_results=/media/fatih/1tb/training_data/nnunet_results

NNU=/home/fatih/nnunet-env/bin
PY=/home/fatih/nnunet-env/bin/python
LOG_DIR=/media/fatih/1tb/training_data/logs
PRED_ROOT=/media/fatih/1tb/training_data/predictions
mkdir -p "$LOG_DIR" "$PRED_ROOT"

ts() { date +%Y%m%d_%H%M%S; }

# (id, name, sigma_for_inference_or_"nogauss")
declare -a VARIANTS=(
    "515:CoronaryPseudoROI_v2:40"
    "521:CoronaryPseudoROIv2Sigma20:20"
    "522:CoronaryPseudoROIv2Sigma30:30"
    "523:CoronaryPseudoROIv2Sigma60:60"
    "524:CoronaryPseudoROIv2Sigma80:80"
    "525:CoronaryPseudoROIv2NoGaussian:nogauss"
)

# ===== 1. build σ ablation datasets =====
echo "================================================================"
echo "[$(ts)] Step 1: build v2 ablation datasets from 515"
echo "================================================================"
"$PY" /media/fatih/1tb/training_data/nnunet_raw/make_v2_ablation.py \
    2>&1 | tee "$LOG_DIR/make_v2_ablation_$(ts).log"

# ===== 2. preprocess =====
for v in "${VARIANTS[@]}"; do
    IFS=':' read -r id name sig <<<"$v"
    echo "================================================================"
    echo "[$(ts)] Step 2: preprocess Dataset${id} (ResEncL)"
    echo "================================================================"
    "$NNU/nnUNetv2_plan_and_preprocess" \
        -d "$id" -pl nnUNetPlannerResEncL -c 2d \
        --verify_dataset_integrity \
        2>&1 | tee "$LOG_DIR/Dataset${id}_preprocess_ResEncL_$(ts).log"
    # Carry splits if 515 has one (so all variants share identical split)
    if [[ -f "$nnUNet_raw/Dataset515_CoronaryPseudoROI_v2/splits_final.json" ]]; then
        cp -f "$nnUNet_raw/Dataset515_CoronaryPseudoROI_v2/splits_final.json" \
              "$nnUNet_preprocessed/Dataset${id}_${name}/splits_final.json"
        echo "  copied splits_final.json from 515"
    fi
done

# ===== 3. training =====
for v in "${VARIANTS[@]}"; do
    IFS=':' read -r id name sig <<<"$v"
    echo "================================================================"
    echo "[$(ts)] Step 3: train Dataset${id} (ResEncL, 50ep, fold 0)"
    echo "================================================================"
    "$NNU/nnUNetv2_train" \
        "$id" 2d 0 \
        -p nnUNetResEncUNetLPlans \
        -tr nnUNetTrainer_50epochs \
        2>&1 | tee "$LOG_DIR/Dataset${id}_fold0_ResEncL_50ep_$(ts).log"
done

# ===== 4. ARCADE test-only inference inputs =====
echo "================================================================"
echo "[$(ts)] Step 4: build ARCADE-test-only σ-matched inference inputs"
echo "================================================================"
"$PY" /media/fatih/1tb/training_data/build_test_inputs_v2.py \
    2>&1 | tee "$LOG_DIR/build_test_inputs_v2_$(ts).log"

# ===== 5. inference =====
for v in "${VARIANTS[@]}"; do
    IFS=':' read -r id name sig <<<"$v"
    out="$PRED_ROOT/Dataset510_test_via${id}_50ep"
    in="$PRED_ROOT/Dataset510_test_inputs/sigma${sig}"
    [[ "$sig" == "nogauss" ]] && in="$PRED_ROOT/Dataset510_test_inputs/nogauss"
    mkdir -p "$out"
    echo "================================================================"
    echo "[$(ts)] Step 5: inference Dataset${id} -> $out (input: $in)"
    echo "================================================================"
    "$NNU/nnUNetv2_predict" \
        -i "$in" -o "$out" \
        -d "$id" -c 2d -f 0 \
        -p nnUNetResEncUNetLPlans \
        -tr nnUNetTrainer_50epochs \
        -chk checkpoint_best.pth \
        2>&1 | tee "$LOG_DIR/Dataset510_test_via${id}_inference_$(ts).log"
done

# ===== 6. comparison =====
echo "================================================================"
echo "[$(ts)] Step 6: comparison report"
echo "================================================================"
"$PY" /media/fatih/1tb/training_data/compare_v2_ablation.py \
    2>&1 | tee "$LOG_DIR/v2_ablation_comparison_$(ts).log"

echo "================================================================"
echo "[$(ts)] v2 ablation pipeline DONE."
echo "================================================================"
