"""Convert ARCADE syntax (train+val+test) into Dataset510_CoronaryARCADE.

Per-annotation ROI extraction matching Dataset507/508 format:
- COCO polygon → binary mask (PIL rasterization)
- Skip stenosis class (cat_id=26) — vessel segments only
- Skip annotations with < MIN_CLASS_PIXELS rasterized pixels
- crop_center = centroid + N(0, sigma=7.5)
- 192x192 crop, zero-padded at boundaries
- Channel 0: angiography (float32, /255)
- Channel 1: 192x192 Gaussian (sigma=40) peaked at original centroid (within crop)
- Label: binary mask of THIS annotation only
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

SRC = Path("/media/fatih/ntfs2/arcade/syntax")
DST = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset510_CoronaryARCADE")

CROP_SIZE = 192
SIGMA_JITTER = 7.5
GAUSSIAN_SIGMA = 40.0
MIN_CLASS_PIXELS = 300
STENOSIS_CAT_ID = 26
SEED = 42

SITK_SPACING = (1.0, 1.0, 1.0)
SITK_ORIGIN = (0.0, 0.0, 0.0)
SITK_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def write_sitk(arr: np.ndarray, path: Path) -> None:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SITK_SPACING)
    img.SetOrigin(SITK_ORIGIN)
    img.SetDirection(SITK_DIRECTION)
    sitk.WriteImage(img, str(path))


def make_gaussian(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def crop_pad(arr: np.ndarray, cy: int, cx: int, size: int, fill: int | float = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    half = size // 2
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size

    pad_top = max(0, -y0)
    pad_bot = max(0, y1 - h)
    pad_lef = max(0, -x0)
    pad_rig = max(0, x1 - w)

    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    cropped = arr[y0c:y1c, x0c:x1c]

    if pad_top or pad_bot or pad_lef or pad_rig:
        pad = ((pad_top, pad_bot), (pad_lef, pad_rig))
        cropped = np.pad(cropped, pad, mode="constant", constant_values=fill)
    return cropped


def rasterize_polygons(polygons: list[list[float]], h: int, w: int) -> np.ndarray:
    """Rasterize COCO polygon list into a binary mask using PIL."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly_flat in polygons:
        if len(poly_flat) < 6:  # need at least 3 (x,y) pairs
            continue
        # PIL expects [(x1,y1), (x2,y2), ...] or flat tuple
        coords = [(poly_flat[i], poly_flat[i + 1]) for i in range(0, len(poly_flat), 2)]
        draw.polygon(coords, fill=1)
    return np.array(img, dtype=np.uint8)


def process_annotation(args):
    """One (image, annotation) pair → one case file or None."""
    img_path, img_array, ann, split = args

    cat_id = ann["category_id"]
    if cat_id == STENOSIS_CAT_ID:
        return None

    seg = ann.get("segmentation")
    if not seg:
        return None

    h, w = img_array.shape
    mask = rasterize_polygons(seg, h, w)
    n_px = int(mask.sum())
    if n_px < MIN_CLASS_PIXELS:
        return None

    ys, xs = np.where(mask > 0)
    cy_c = float(ys.mean())
    cx_c = float(xs.mean())

    # Deterministic per-annotation jitter
    rng = np.random.default_rng(abs(hash((SEED, split, ann["image_id"], ann["id"]))) % (2**32))
    cy = int(round(cy_c + rng.normal(0.0, SIGMA_JITTER)))
    cx = int(round(cx_c + rng.normal(0.0, SIGMA_JITTER)))

    img_crop = crop_pad(img_array, cy, cx, CROP_SIZE, fill=0).astype(np.float32) / 255.0
    lbl_crop = crop_pad(mask, cy, cx, CROP_SIZE, fill=0).astype(np.uint8)

    half = CROP_SIZE // 2
    peak_y = half + (cy_c - cy)
    peak_x = half + (cx_c - cx)
    gauss = make_gaussian(CROP_SIZE, CROP_SIZE, peak_y, peak_x, GAUSSIAN_SIGMA)

    case_id = f"arcade_{split}_img{ann['image_id']:04d}_c{cat_id:02d}_a{ann['id']:05d}"
    write_sitk(img_crop, DST / "imagesTr" / f"{case_id}_0000.nii.gz")
    write_sitk(gauss,    DST / "imagesTr" / f"{case_id}_0001.nii.gz")
    write_sitk(lbl_crop, DST / "labelsTr" / f"{case_id}.nii.gz")
    return case_id


def collect_tasks() -> list:
    tasks = []
    for split in ("train", "val", "test"):
        img_dir = SRC / split / "images"
        ann_file = SRC / split / "annotations" / f"{split}.json"
        if not ann_file.exists():
            print(f"  skip {split}: no {ann_file}")
            continue
        coco = json.load(open(ann_file))
        images = {img["id"]: img for img in coco["images"]}
        # Group annotations by image
        for ann in coco["annotations"]:
            img_meta = images.get(ann["image_id"])
            if img_meta is None:
                continue
            img_path = img_dir / img_meta["file_name"]
            if not img_path.exists():
                continue
            tasks.append((img_path, ann, split))
    return tasks


def _worker(args):
    img_path, ann, split = args
    img_array = np.array(Image.open(img_path), dtype=np.uint8)
    if img_array.ndim != 2:
        img_array = img_array[..., 0]  # take first channel if RGB
    return process_annotation((img_path, img_array, ann, split))


def main() -> None:
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks()
    print(f"Annotations to process: {len(tasks)}")

    n_done = 0
    n_written = 0
    with ProcessPoolExecutor(max_workers=24) as ex:
        for case_id in ex.map(_worker, tasks, chunksize=64):
            n_done += 1
            if case_id is not None:
                n_written += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(tasks)} annotations | {n_written} cases written")

    print(f"\nDONE: {n_done} annotations processed, {n_written} cases written")

    ds_json = {
        "channel_names": {"0": "angiography", "1": "roi_gaussian"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": n_written,
        "file_ending": ".nii.gz",
        "name": "CoronaryARCADE",
        "description": (
            "Test set built from the ARCADE syntax subset (train+val+test). "
            "Per-annotation 192x192 ROI crops centered on the segmentation centroid "
            f"with N(0, sigma={SIGMA_JITTER}) jitter. Channel 1 is a Gaussian "
            f"(sigma={GAUSSIAN_SIGMA}) peaked at the centroid. Stenosis class skipped. "
            "Format matches Dataset508 — can be used directly with nnUNetv2_predict."
        ),
        "tensorImageSize": "2D",
        "reference": "ARCADE dataset (Popov et al.)",
        "licence": "see ARCADE license",
        "release": "2026-05-18",
        "roi_strategy": "per_annotation_centroid_with_gaussian",
        "crop_size": CROP_SIZE,
        "base_sigma": GAUSSIAN_SIGMA,
        "jitter_sigma": SIGMA_JITTER,
        "min_class_pixels": MIN_CLASS_PIXELS,
        "source_subset": "syntax (train+val+test=1500 images)",
        "skipped_classes": [STENOSIS_CAT_ID],
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(ds_json, f, indent=2)
    print(f"dataset.json written ({n_written} cases)")


if __name__ == "__main__":
    main()
