"""Evaluate a multi-class teacher model trained on Dataset520 (ARCADE syntax) on
the 300-image held-out test set. Computes:

  * Per-class binary Dice for each of the 25 vessel classes
  * Macro-mean per-class Dice (mean over classes with any GT presence)
  * Binary-collapsed Dice (all classes 1..25 -> 1) — matches Zhang et al. 2025
  * clDice (centerline Dice, binary-collapsed)
  * HD95 (binary-collapsed)
  * Precision / Recall (binary-collapsed)

Usage:
    python eval_dataset520_multiclass.py <pred_dir>
        — defaults to predictions/Dataset520_test_predictions/
GT directory: nnunet_raw/Dataset520_CoronaryARCADEMultiClass/labelsTs/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import hd95
from skimage.morphology import skeletonize

PRED_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/media/fatih/1tb/training_data/predictions/Dataset520_test_predictions"
)
GT_DIR = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset520_CoronaryARCADEMultiClass/labelsTs")
N_CLASSES = 25


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return float("nan")
    return float(2 * np.logical_and(pred, gt).sum() / s)


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Topology-aware Dice based on skeletons (Shit et al., 2021)."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return float("nan")
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)
    tprec = (skel_p & gt).sum() / max(skel_p.sum(), 1)
    tsens = (skel_g & pred).sum() / max(skel_g.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return float(p), float(r)


def hd95_safe(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    return float(hd95(pred, gt))


def read(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).squeeze().astype(np.uint8)


def main() -> None:
    pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
    print(f"predictions: {len(pred_files)} in {PRED_DIR}")

    if not pred_files:
        print("no predictions found"); return

    per_class_d: dict[int, list[float]] = defaultdict(list)
    bin_d, cld, hd, prec, rec = [], [], [], [], []
    missing_gt = 0

    for i, p in enumerate(pred_files):
        gt_p = GT_DIR / p.name
        if not gt_p.exists():
            missing_gt += 1
            continue
        pred = read(p)
        gt = read(gt_p)

        for c in range(1, N_CLASSES + 1):
            if (gt == c).sum() == 0:
                continue
            d = dice(pred == c, gt == c)
            if not np.isnan(d):
                per_class_d[c].append(d)

        bp = pred > 0
        bg = gt > 0
        bin_d.append(dice(bp, bg))
        cld.append(cl_dice(bp, bg))
        hd.append(hd95_safe(bp, bg))
        pp, rr = precision_recall(bp, bg)
        prec.append(pp); rec.append(rr)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pred_files)}")

    if missing_gt:
        print(f"WARNING: {missing_gt} predictions had no matching GT")

    bin_arr = np.array([x for x in bin_d if not np.isnan(x)])
    cld_arr = np.array([x for x in cld if not np.isnan(x)])
    hd_arr = np.array([x for x in hd if not np.isnan(x)])
    prec_arr = np.array([x for x in prec if not np.isnan(x)])
    rec_arr = np.array([x for x in rec if not np.isnan(x)])

    macro = np.mean([np.mean(v) for v in per_class_d.values() if v])

    print("\n" + "=" * 70)
    print(f"BINARY-COLLAPSED (vs Zhang et al. 2025 metric)")
    print("=" * 70)
    print(f"  Dice       : {bin_arr.mean():.4f}  ± {bin_arr.std():.4f}   (Zhang 0.7674)")
    print(f"  Precision  : {prec_arr.mean():.4f}  ± {prec_arr.std():.4f}   (Zhang 0.8066)")
    print(f"  Recall     : {rec_arr.mean():.4f}  ± {rec_arr.std():.4f}   (Zhang 0.7487)")
    print(f"  clDice     : {cld_arr.mean():.4f}  ± {cld_arr.std():.4f}   (Zhang 0.5030)")
    print(f"  HD95 (px)  : {hd_arr.mean():.4f}  ± {hd_arr.std():.4f}   (Zhang 57.84)")

    print("\n" + "=" * 70)
    print(f"PER-CLASS DICE (mean over images containing the class)")
    print("=" * 70)
    print(f"  {'cls':4s} {'n':>4} {'dice':>7}")
    for c in sorted(per_class_d):
        v = per_class_d[c]
        print(f"  c{c:02d}  {len(v):>4} {np.mean(v):>7.4f}")
    print(f"\n  macro-mean per-class dice: {macro:.4f}")

    report = {
        "pred_dir": str(PRED_DIR),
        "n_test": len(bin_d),
        "binary_collapsed": {
            "dice_mean": float(bin_arr.mean()),
            "dice_std": float(bin_arr.std()),
            "precision_mean": float(prec_arr.mean()),
            "recall_mean": float(rec_arr.mean()),
            "clDice_mean": float(cld_arr.mean()),
            "hd95_mean_px": float(hd_arr.mean()),
        },
        "per_class": {f"c{c:02d}": {
            "n": len(per_class_d[c]),
            "dice_mean": float(np.mean(per_class_d[c])),
        } for c in sorted(per_class_d)},
        "macro_per_class_dice": float(macro),
        "zhang_reference": {
            "Dice": 0.7674, "Precision": 0.8066, "Recall": 0.7487,
            "clDice": 0.5030, "HD95": 57.84,
        },
    }
    out = PRED_DIR / "dataset520_eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
