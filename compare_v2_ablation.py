"""Comparison report for v2 sigma ablation on ARCADE test 300 (per-annotation
subset of Dataset510 with case_id prefix 'arcade_test_').

Reads predictions from:
    predictions/Dataset510_test_via{515,521,522,523,524,525}_50ep/
Ground truth:
    nnunet_raw/Dataset510_CoronaryARCADE/labelsTr/<case>.nii.gz   (arcade_test_*)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PRED_ROOT = Path("/media/fatih/1tb/training_data/predictions")
GT_DIR = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset510_CoronaryARCADE/labelsTr")
OUT_DIR = PRED_ROOT

MODELS = [
    ("515", "Dataset510_test_via515_50ep", "σ=40 (main, Dataset515)"),
    ("521", "Dataset510_test_via521_50ep", "σ=20"),
    ("522", "Dataset510_test_via522_50ep", "σ=30"),
    ("523", "Dataset510_test_via523_50ep", "σ=60"),
    ("525", "Dataset510_test_via525_50ep", "no-gaussian"),
]

CLS_RE = re.compile(r"arcade_test_img\d+_c(\d+)_a\d+")


def dice(p, g):
    p = p.astype(bool); g = g.astype(bool)
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(p, g).sum() / s)


def score_dir(pred_dir: Path) -> dict:
    out = {}
    for p in sorted(pred_dir.glob("arcade_test_*.nii.gz")):
        gt_p = GT_DIR / p.name
        if not gt_p.exists():
            continue
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.uint8)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_p))).astype(np.uint8)
        d = dice(pred, gt)
        m = CLS_RE.search(p.name)
        cls = f"c{int(m.group(1)):02d}" if m else "?"
        out[p.stem.replace(".nii", "")] = (d, cls, int(pred.sum()), int(gt.sum()))
    return out


def summarize(per_case, desc):
    vals = np.array([v[0] for v in per_case.values()])
    by_cls = defaultdict(list)
    for d, cl, _, _ in per_case.values():
        by_cls[cl].append(d)
    return {
        "desc": desc, "n": len(vals),
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "median": float(np.median(vals)) if len(vals) else float("nan"),
        "std": float(vals.std()) if len(vals) else float("nan"),
        "empty_pred": sum(1 for v in per_case.values() if v[2] == 0),
        "by_class": {cl: {"n": len(a), "mean": float(np.mean(a))} for cl, a in by_cls.items()},
    }


def main():
    scored = {}
    for key, folder, desc in MODELS:
        d = PRED_ROOT / folder
        if not d.exists():
            print(f"  [skip] {key}: {folder} missing"); continue
        n = len(list(d.glob("arcade_test_*.nii.gz")))
        print(f"  scoring {key} ({n} files) — {desc}")
        scored[key] = {"per_case": score_dir(d), "desc": desc}

    if not scored:
        print("no models scored"); return

    common = set.intersection(*[set(s["per_case"]) for s in scored.values()])
    print(f"\ncommon arcade_test_ cases across all models: {len(common)}")

    summaries = {}
    for k, s in scored.items():
        sub = {c: s["per_case"][c] for c in common}
        summaries[k] = summarize(sub, s["desc"])

    keys = list(scored.keys())

    print("\n" + "=" * 80)
    print("OVERALL — ARCADE held-out test (arcade_test_*) per-annotation Dice")
    print("=" * 80)
    print(f"  {'id':4s} {'desc':28s} {'n':>5} {'mean':>8} {'median':>8} {'std':>8} {'empty':>6}")
    for k, s in sorted([(k, summaries[k]) for k in keys], key=lambda r: -r[1]["mean"]):
        print(f"  {k:4s} {s['desc']:28s} {s['n']:>5d} {s['mean']:>8.4f} {s['median']:>8.4f} {s['std']:>8.4f} {s['empty_pred']:>6d}")

    print("\n" + "=" * 80)
    print("BY CLASS (mean dice)")
    print("=" * 80)
    classes = sorted({cl for k in keys for cl in summaries[k]["by_class"]})
    header = f"  {'cls':4s} {'n':>5} " + " ".join(f"{k:>9}" for k in keys) + "   best"
    print(header)
    for cl in classes:
        n = summaries[keys[0]]["by_class"].get(cl, {}).get("n", 0)
        means = {k: summaries[k]["by_class"].get(cl, {}).get("mean") for k in keys}
        best = max(means, key=lambda k: means[k] if means[k] is not None else -1)
        vals = " ".join(f"{means[k]:>9.4f}" if means[k] is not None else f"{'-':>9}" for k in keys)
        print(f"  {cl} {n:>5d} {vals}   {best}")

    # Head-to-head (each model vs no-gaussian and vs main 515)
    print("\n" + "=" * 80)
    print("Δ vs 525 (no-gaussian) and vs 515 (σ=40 main)")
    print("=" * 80)
    for ref in ("525", "515"):
        if ref not in summaries: continue
        print(f"\n  vs {ref} ({summaries[ref]['desc']}, mean={summaries[ref]['mean']:.4f}):")
        for k in keys:
            if k == ref: continue
            d = summaries[k]["mean"] - summaries[ref]["mean"]
            print(f"    {k} ({summaries[k]['desc']:25s}): mean={summaries[k]['mean']:.4f}  Δ={d:+.4f}")

    # Per-case head-to-head wins
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD per-case wins (|Δ| > 1e-4)")
    print("=" * 80)
    EPS = 1e-4
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            wa = wb = tie = 0; sumd = 0.0
            for c in common:
                da = scored[a]["per_case"][c][0]; db = scored[b]["per_case"][c][0]
                sumd += (da - db)
                if abs(da - db) < EPS: tie += 1
                elif da > db: wa += 1
                else: wb += 1
            print(f"  {a} vs {b}: {a}={wa:4d}  {b}={wb:4d}  ties={tie:3d}  mean Δ={sumd/len(common):+.4f}")

    report = {
        "n_common": len(common),
        "test_subset": "arcade_test_* (300 ARCADE syntax test images, per-annotation crops)",
        "models": {k: summaries[k] for k in keys},
    }
    (OUT_DIR / "v2_ablation_comparison.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_DIR/'v2_ablation_comparison.json'}")

    csv = OUT_DIR / "v2_ablation_per_case_dice.csv"
    with csv.open("w") as f:
        f.write("case,class," + ",".join(f"dice_{k}" for k in keys) + "\n")
        for c in sorted(common):
            cl = scored[keys[0]]["per_case"][c][1]
            dices = [f"{scored[k]['per_case'][c][0]:.4f}" for k in keys]
            f.write(f"{c},{cl}," + ",".join(dices) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
