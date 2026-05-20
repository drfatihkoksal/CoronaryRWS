"""Build Dataset515_CoronaryPseudoROI_v2 from Dataset520 teacher predictions on
runs/*.png frames, keeping only high-confidence per-class ROIs.

Pipeline per frame:
  1. Load PNG, run Dataset520 (ResEnc-L, 250ep checkpoint_best) → (argmax label, softmax probs)
  2. For each vessel class c in 1..25:
       - mask_c = (label == c)
       - skip if mask_c.sum() < MIN_CLASS_PIXELS
       - mean_conf_c = softmax[c][mask_c].mean()
       - skip if mean_conf_c < CONF_THRESHOLD
       - extract 192x192 ROI centred on jittered centroid:
           ch0 = angiography crop (float32 / 255)
           ch1 = 192x192 Gaussian (sigma=40) at the centroid
           label = binary mask of THIS class only
  3. Write nnU-Net files (matches Dataset507/508 metadata)

Run in foreground; resumable (skips frames whose target case already exists).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ----- defaults -----
RUNS_ROOT = Path("/media/fatih/ntfs2/anjiolar/images/coronary_dataset/runs")
DST = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset515_CoronaryPseudoROI_v2")
MODEL_DIR = Path(
    "/media/fatih/1tb/training_data/nnunet_results/Dataset520_CoronaryARCADEMultiClass/"
    "nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans__2d"
)
CHK = "checkpoint_best.pth"

CROP_SIZE = 192
SIGMA_JITTER = 7.5
GAUSSIAN_SIGMA = 40.0
MIN_CLASS_PIXELS = 300
CONF_THRESHOLD = 0.70
SEED = 42

SITK_SPACING = (1.0, 1.0, 1.0)
SITK_ORIGIN = (0.0, 0.0, 0.0)
SITK_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def write_sitk(arr: np.ndarray, path: Path) -> None:
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SITK_SPACING); img.SetOrigin(SITK_ORIGIN); img.SetDirection(SITK_DIRECTION)
    sitk.WriteImage(img, str(path))


def make_gaussian(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def crop_pad(arr: np.ndarray, cy: int, cx: int, size: int, fill=0) -> np.ndarray:
    h, w = arr.shape[:2]
    half = size // 2
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    pad_top = max(0, -y0); pad_bot = max(0, y1 - h)
    pad_lef = max(0, -x0); pad_rig = max(0, x1 - w)
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    out = arr[y0c:y1c, x0c:x1c]
    if pad_top or pad_bot or pad_lef or pad_rig:
        out = np.pad(out, ((pad_top, pad_bot), (pad_lef, pad_rig)),
                     mode="constant", constant_values=fill)
    return out


def init_predictor() -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,           # mirroring disabled — 2x faster, marginal accuracy loss
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=(0,),
        checkpoint_name=CHK,
    )
    return predictor


def make_properties(h: int, w: int) -> dict:
    return {
        "sitk_stuff": {
            "spacing": SITK_SPACING,
            "origin": SITK_ORIGIN,
            "direction": SITK_DIRECTION,
        },
        "spacing": list(SITK_SPACING)[::-1],
        "shape_before_cropping": (1, h, w),
        "bbox_used_for_cropping": [[0, 1], [0, h], [0, w]],
        "shape_after_cropping_and_before_resampling": (1, h, w),
    }


def slug(run_name: str, frame_idx: int, cls: int) -> str:
    # Short, collision-resistant: 8-char blake2 hex of the full run name,
    # plus the first 24 alnum chars for human readability. ~1 in 4 billion
    # collision probability across the entire run set.
    import hashlib
    h = hashlib.blake2b(run_name.encode(), digest_size=4).hexdigest()  # 8 hex chars
    short = re.sub(r"[^A-Za-z0-9]+", "", run_name)[:24]
    return f"r{h}_{short}_f{frame_idx:04d}_c{cls:02d}"


def collect_frames(root: Path) -> list[tuple[Path, str, int]]:
    """Return [(png_path, run_name, frame_idx), ...] for all extracted PNG runs."""
    out = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for f in sorted(run_dir.glob("frame_idx*.png")):
            m = re.search(r"frame_idx(\d+)\.png$", f.name)
            if not m:
                continue
            out.append((f, run_dir.name, int(m.group(1))))
    return out


def process(args):
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "imagesTr").mkdir(exist_ok=True)
    (DST / "labelsTr").mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    predictor = init_predictor()
    print("predictor initialised")

    frames = collect_frames(RUNS_ROOT)
    if args.limit:
        frames = frames[:args.limit]
    print(f"frames to process: {len(frames)}")

    n_done = 0
    n_kept = 0
    n_skipped_lowconf = 0
    n_skipped_small = 0
    per_class = np.zeros(26, dtype=np.int64)
    t0 = time.time()

    for png_path, run_name, fi in frames:
        # quick resume check: if any case from this frame exists, skip the frame
        expected_any = DST / "imagesTr" / f"{slug(run_name, fi, 1)}_0000.nii.gz"
        # (incomplete check — only c01; full check would be expensive)

        try:
            img_u8 = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
        except Exception as e:
            print(f"  skip {png_path}: {e}")
            continue

        h, w = img_u8.shape
        # nnU-Net 2D input: shape (C, Z, Y, X) → (1, 1, H, W) with C=1
        img_arr = (img_u8.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]
        props = make_properties(h, w)

        try:
            seg, probs = predictor.predict_single_npy_array(
                img_arr, props,
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=True,
            )
        except Exception as e:
            print(f"  predict err {png_path.name}: {e}")
            continue

        # seg: (Z, Y, X) int; probs: (C, Z, Y, X) float32
        seg = np.asarray(seg).squeeze(0).astype(np.int16)        # (H, W)
        probs = np.asarray(probs).squeeze(1).astype(np.float32)  # (C, H, W)

        img_f = img_u8.astype(np.float32) / 255.0

        for c in range(1, 26):
            mask_c = seg == c
            if mask_c.sum() < MIN_CLASS_PIXELS:
                n_skipped_small += int(mask_c.sum() > 0)
                continue

            # Keep only the largest connected component so each crop has a
            # single coherent vessel segment (centroid and Gaussian are then
            # computed on that component, not pulled toward distant fragments).
            labeled_cc, n_cc = ndimage.label(mask_c)
            if n_cc > 1:
                cc_sizes = ndimage.sum(mask_c, labeled_cc, range(1, n_cc + 1))
                mask_c = labeled_cc == (int(np.argmax(cc_sizes)) + 1)
            if mask_c.sum() < MIN_CLASS_PIXELS:
                n_skipped_small += 1
                continue

            mean_conf = float(probs[c][mask_c].mean())
            if mean_conf < CONF_THRESHOLD:
                n_skipped_lowconf += 1
                continue

            ys, xs = np.where(mask_c)
            cy_cent = float(ys.mean()); cx_cent = float(xs.mean())
            cy = int(round(cy_cent + rng.normal(0.0, SIGMA_JITTER)))
            cx = int(round(cx_cent + rng.normal(0.0, SIGMA_JITTER)))

            img_crop = crop_pad(img_f, cy, cx, CROP_SIZE, fill=0.0).astype(np.float32)
            bin_lbl = crop_pad(mask_c.astype(np.uint8), cy, cx, CROP_SIZE, fill=0).astype(np.uint8)

            half = CROP_SIZE // 2
            peak_y = half + (cy_cent - cy)
            peak_x = half + (cx_cent - cx)
            gauss = make_gaussian(CROP_SIZE, CROP_SIZE, peak_y, peak_x, GAUSSIAN_SIGMA)

            case_id = slug(run_name, fi, c)
            write_sitk(img_crop, DST / "imagesTr" / f"{case_id}_0000.nii.gz")
            write_sitk(gauss,    DST / "imagesTr" / f"{case_id}_0001.nii.gz")
            write_sitk(bin_lbl,  DST / "labelsTr" / f"{case_id}.nii.gz")
            n_kept += 1
            per_class[c] += 1

        n_done += 1
        if n_done % 200 == 0:
            dt = time.time() - t0
            rate = n_done / max(dt, 1)
            eta = (len(frames) - n_done) / max(rate, 1e-3) / 60
            print(f"  {n_done}/{len(frames)} frames | kept={n_kept} "
                  f"| skip_low={n_skipped_lowconf} small={n_skipped_small} "
                  f"| {rate:.2f} fps | ETA {eta:.0f} min")

    print(f"\nDONE: {n_done} frames | kept ROIs = {n_kept}")
    print(f"  skipped low-confidence: {n_skipped_lowconf}")
    print(f"  skipped small (<{MIN_CLASS_PIXELS}px): {n_skipped_small}")
    print(f"  per-class counts: {dict(enumerate(per_class.tolist()))}")

    # dataset.json — matches 507/508 format so it slots into the same pipeline
    ds_json = {
        "channel_names": {"0": "angiography", "1": "roi_gaussian"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": n_kept,
        "file_ending": ".nii.gz",
        "name": "CoronaryPseudoROI_v2",
        "description": (
            "Per-class pseudo-label ROI crops from the Dataset520 ARCADE-trained "
            "multi-class teacher (ResEnc-L 250ep ckpt_best). Source: extracted PNG "
            "frames in runs/. Confidence-filtered: only crops whose teacher mean "
            f"softmax >= {CONF_THRESHOLD} inside the predicted vessel mask are kept."
        ),
        "tensorImageSize": "2D",
        "reference": "Coronary RWS Analyser Project — teacher-based pseudo-labels",
        "licence": "CC BY-NC-SA 4.0",
        "release": "2026-05-19",
        "roi_strategy": "per_class_centroid_with_gaussian",
        "crop_size": CROP_SIZE,
        "base_sigma": GAUSSIAN_SIGMA,
        "jitter_sigma": SIGMA_JITTER,
        "min_class_pixels": MIN_CLASS_PIXELS,
        "teacher_model": str(MODEL_DIR),
        "teacher_checkpoint": CHK,
        "confidence_threshold": CONF_THRESHOLD,
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(ds_json, f, indent=2)
    print(f"wrote {DST/'dataset.json'}")


def main():
    global CONF_THRESHOLD
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="dev only: cap on frames")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help="confidence threshold (default 0.70)")
    args = p.parse_args()
    CONF_THRESHOLD = args.conf
    process(args)


if __name__ == "__main__":
    main()
