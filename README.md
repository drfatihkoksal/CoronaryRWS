# CoronaryRWS — ROI-Guided Coronary Vessel Segmentation Pipeline

Code for the paper:
> **ROI-Guided Dual-Channel nnU-Net with Gaussian Spatial Attention for Coronary Vessel Segmentation: Teacher–Student Pseudo-Label Pipeline and Gaussian Width Ablation**
> Fatih Köksal et al., submitted to *International Journal of Imaging Systems and Technology* (Wiley), 2026.

---

## Overview

This repository contains the full reproducibility pipeline:

1. **Dataset construction** — ARCADE annotations → nnU-Net datasets (Dataset510, Dataset520)
2. **Teacher training** — ResEnc-L multi-class nnU-Net on 1,200 ARCADE trainval images (Dataset520)
3. **Pseudo-label generation** — teacher inference on ~61k institutional frames → Dataset515
4. **σ-ablation** — power-transform shortcut to derive Datasets 521–525 from Dataset515
5. **Training & inference** — 50-epoch ResEnc-L students, inference on ARCADE held-out test
6. **Evaluation** — per-annotation Dice, per-SYNTAX-class breakdown, head-to-head win rates

---

## Requirements

```bash
# nnU-Net v2 (with ResEnc-L planner)
pip install nnunetv2

# Additional packages
pip install SimpleITK pydicom pylibjpeg[all] medpy blosc2
```

Set environment variables before running any nnU-Net command:
```bash
export nnUNet_raw=/path/to/nnunet_raw
export nnUNet_preprocessed=/path/to/nnunet_preprocessed
export nnUNet_results=/path/to/nnunet_results
```

---

## Dataset Construction

### ARCADE held-out test bank (Dataset510)
```bash
python nnunet_raw/make_dataset510_arcade.py
```
Converts ARCADE COCO annotations to per-annotation 192×192 ROI crops (2-channel: angiography + Gaussian σ=40).

### ARCADE multi-class teacher dataset (Dataset520)
```bash
python nnunet_raw/make_dataset520_arcade_multiclass.py
```
Converts ARCADE COCO annotations to 512×512 whole-image 25-class segmentation maps.

### σ-ablation family (Datasets 521–525) from Dataset515
```bash
python nnunet_raw/make_v2_ablation.py
```
Derives σ-variant datasets from Dataset515 via the exact power-transform shortcut:
`G_σ' = G_σ0 ^ (σ0² / σ'²)` — no re-preprocessing of raw frames needed.

---

## Teacher Training (Dataset520)

```bash
nnUNetv2_plan_and_preprocess -d 520 -pl nnUNetPlannerResEncL -c 2d --verify_dataset_integrity
nnUNetv2_train 520 2d 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs
```
Best checkpoint: epoch 42 (EMA validation Dice). Training was stopped at epoch 51.

### Evaluate teacher on ARCADE held-out test (300 images)
```bash
nnUNetv2_predict -i predictions/Dataset510_test_inputs/sigma40 \
    -o predictions/Dataset520_test_predictions \
    -d 520 -c 2d -f 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs \
    -chk checkpoint_best.pth

python eval_dataset520_multiclass.py
```

---

## Pseudo-Label Generation (Dataset515)

```bash
python build_dataset515_pseudo.py
```
Runs the Dataset520 teacher on all institutional frames. For each predicted vessel class with ≥300 pixels and mean softmax ≥0.70, extracts a 192×192 ROI crop. Produces Dataset515 (~209,216 crops).

---

## Full σ-Ablation Pipeline

```bash
bash run_v2_ablation_pipeline.sh
```

This script runs the complete pipeline: dataset derivation → preprocessing → 50-epoch training for Datasets 515, 521, 522, 523, 525 → inference on ARCADE held-out test → evaluation report.

Or step by step:
```bash
# 1. Build σ-variant inputs for the test set
python build_test_inputs_v2.py

# 2. Preprocess + train Datasets 521, 522, 523, 525
bash preprocess_and_train_521_525.sh

# 3. Compare all variants
python compare_v2_ablation.py
# → predictions/v2_ablation_comparison.json
# → predictions/v2_ablation_per_case_dice.csv
```

---

## Results Summary

| Dataset | σ (px) | Mean Dice | Median Dice |
|---------|--------|-----------|-------------|
| DS515   | 40 (ref) | **0.8002** | **0.8654** |
| DS521   | 20     | 0.8001    | 0.8650      |
| DS523   | 60     | 0.7998    | 0.8653      |
| DS522   | 30     | 0.7983    | 0.8638      |
| DS525   | no-Gaussian | 0.7223 | 0.8032   |

The Gaussian attention channel provides +7.8 pp mean Dice over the no-Gaussian baseline; performance is insensitive to σ over the range 20–60 px (max pairwise difference: 0.002 Dice).

---

## File Structure

```
nnunet_raw/
  make_dataset510_arcade.py          # ARCADE COCO → Dataset510 (per-annotation ROI crops)
  make_dataset520_arcade_multiclass.py  # ARCADE COCO → Dataset520 (whole-image multi-class)
  make_v2_ablation.py                # Dataset515 → Datasets 521/522/523/524/525

build_dataset515_pseudo.py           # Teacher inference + pseudo-label ROI extraction
build_test_inputs_v2.py              # σ-matched test inputs from Dataset510 arcade_test_*
eval_dataset520_multiclass.py        # Teacher evaluation (Dice, clDice, HD95, precision)
compare_v2_ablation.py               # σ-ablation comparison report

run_v2_ablation_pipeline.sh          # Full pipeline: derive → preprocess → train → eval
preprocess_and_train_521_525.sh      # Preprocess + train σ-ablation variants

paper_ijist/manuscript.tex           # Manuscript source (LaTeX)
```

---

## Citation

```bibtex
@article{koksal2026coronary,
  title   = {ROI-Guided Dual-Channel nnU-Net with Gaussian Spatial Attention
             for Coronary Vessel Segmentation},
  author  = {K{\"o}ksal, Fatih and Levent, Fatih and Koca, Fatih and
             Severg{\"u}n, K{\"u}bra and Melek, Mehmet and
             Vatansever A\u{g}ca, Fahriye and Tenekecio\u{g}lu, Erhan and
             Ar{\i}, Hasan},
  journal = {International Journal of Imaging Systems and Technology},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Ethics

Retrospective use of institutional angiogram data was approved by the local ethics committee (approval no. 2024-TBEK 2026/03-19). No patient-identifiable data is included in this repository.
