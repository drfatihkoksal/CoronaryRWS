"""Convert ARCADE syntax → Dataset520_CoronaryARCADEMultiClass (whole-image multi-class).

Pipeline:
- syntax/train (1000) + syntax/val (200) → imagesTr/labelsTr  (1200 trainval cases)
- syntax/test (300) → imagesTs/labelsTs  (held out for final evaluation)
- 25 SYNTAX vessel classes (cat 1-25) rasterised as multi-class label map
- Overlap rule: bigger-area-first, smaller-area-last → small/distal vessels win
- 512x512 grayscale (.nii.gz, float32 [0,1])
- Stenosis (cat 26) not present in syntax subset, so nothing to skip

Held-out test split matches Zhang et al. 2025 (Bioengineering, ARCADE 1000/200/300).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

SRC = Path("/media/fatih/ntfs2/arcade/syntax")
DST = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset520_CoronaryARCADEMultiClass")
N_CLASSES = 25

SITK_SPACING = (1.0, 1.0, 1.0)
SITK_ORIGIN = (0.0, 0.0, 0.0)
SITK_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

# SYNTAX class names (category_id → anatomical label from ARCADE COCO categories)
SYNTAX_NAMES = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "9a", 11: "10", 12: "10a", 13: "11", 14: "12",
    15: "12a", 16: "13", 17: "14", 18: "14a", 19: "15", 20: "16",
    21: "16a", 22: "16b", 23: "16c", 24: "12b", 25: "14b",
}


def write_sitk(arr: np.ndarray, path: Path) -> None:
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SITK_SPACING)
    img.SetOrigin(SITK_ORIGIN)
    img.SetDirection(SITK_DIRECTION)
    sitk.WriteImage(img, str(path))


def rasterize(anns: list, h: int, w: int) -> np.ndarray:
    """Multi-class label: bigger area first, smaller wins on overlap."""
    label = np.zeros((h, w), dtype=np.uint8)
    anns_sorted = sorted(anns, key=lambda a: -float(a["area"]))
    for ann in anns_sorted:
        cat = int(ann["category_id"])
        if cat < 1 or cat > N_CLASSES:
            continue
        for poly in ann["segmentation"]:
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).polygon(poly, fill=cat)
            arr = np.asarray(mask, dtype=np.uint8)
            label[arr == cat] = cat
    return label


def process_one(args):
    img_path, anns, h, w, out_img, out_lbl = args
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    lbl = rasterize(anns, h, w)
    write_sitk(img.astype(np.float32) / 255.0, out_img)
    write_sitk(lbl, out_lbl)


def process_split(split: str, dst_img: Path, dst_lbl: Path, workers: int) -> list[str]:
    coco_path = SRC / split / "annotations" / f"{split}.json"
    with open(coco_path) as f:
        coco = json.load(f)
    images = {int(im["id"]): im for im in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    tasks = []
    case_ids = []
    for im_id in sorted(images):
        im = images[im_id]
        case_id = f"arcade_{split}_img{im_id:04d}"
        case_ids.append(case_id)
        tasks.append((
            SRC / split / "images" / im["file_name"],
            anns_by_img.get(im_id, []),
            int(im["height"]), int(im["width"]),
            dst_img / f"{case_id}_0000.nii.gz",
            dst_lbl / f"{case_id}.nii.gz",
        ))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(process_one, tasks, chunksize=8)):
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i+1}/{len(tasks)}")
    return case_ids


def main() -> None:
    import os
    workers = max(1, (os.cpu_count() or 4) - 2)

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    (DST / "imagesTs").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTs").mkdir(parents=True, exist_ok=True)

    print("processing train…")
    tr1 = process_split("train", DST / "imagesTr", DST / "labelsTr", workers)
    print("processing val…")
    tr2 = process_split("val", DST / "imagesTr", DST / "labelsTr", workers)
    print("processing test (held out)…")
    ts = process_split("test", DST / "imagesTs", DST / "labelsTs", workers)

    tr_cases = tr1 + tr2
    print(f"\ntrainval: {len(tr_cases)}  | held-out test: {len(ts)}")

    labels = {"background": 0}
    for c in range(1, N_CLASSES + 1):
        labels[f"c{c:02d}_{SYNTAX_NAMES[c]}"] = c

    ds_json = {
        "channel_names": {"0": "angiography"},
        "labels": labels,
        "numTraining": len(tr_cases),
        "file_ending": ".nii.gz",
        "name": "CoronaryARCADEMultiClass",
        "description": (
            "ARCADE syntax 1000+200 trainval + 300 held-out test (matches Zhang et al. 2025 split). "
            "Whole 512x512 grayscale frames with multi-class polygon rasterisation of the 25 "
            "SYNTAX vessel categories. Overlap rule: larger area first, smaller area wins. "
            "Stenosis (cat 26) absent from the syntax subset. Built to train a whole-image teacher "
            "for downstream per-class pseudo-labelling of hospital coronary frames."
        ),
        "tensorImageSize": "2D",
        "reference": "ARCADE — Popov et al., Sci Data 2024",
        "licence": "see ARCADE license",
        "release": "2026-05-19",
        "image_size": 512,
        "overlap_rule": "larger-area-first; smaller-area wins (distal preserved)",
        "source_subset": "syntax",
        "n_vessel_classes": N_CLASSES,
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(ds_json, f, indent=2)

    # Single-fold split: nnU-Net's auto-generated 5-fold runs on the 1200 trainval.
    # No separate splits_final.json — let nnU-Net seed/shuffle it itself.
    print(f"wrote dataset.json ({len(tr_cases)} training, {len(ts)} held-out test)")


if __name__ == "__main__":
    main()
