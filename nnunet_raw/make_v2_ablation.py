"""v2 sigma ablation of Dataset515 (the new teacher-pseudo-label main dataset).

Creates:
    521 = sigma 20  (power transform 515 ch1)
    522 = sigma 30  (power transform 515 ch1)
    523 = sigma 60  (power transform 515 ch1)
    524 = sigma 80  (power transform 515 ch1)
    525 = no-gaussian (drop ch1 entirely)

ch0 (angiography) and labels are hardlinked from Dataset515 so disk cost is
ch1 only (or zero for 525).
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW = Path("/media/fatih/1tb/training_data/nnunet_raw")
SRC_DS = RAW / "Dataset515_CoronaryPseudoROI_v2"
BASE_SIGMA = 40.0

SIGMA_TARGETS = [
    (521, 20.0, "CoronaryPseudoROIv2Sigma20"),
    (522, 30.0, "CoronaryPseudoROIv2Sigma30"),
    (523, 60.0, "CoronaryPseudoROIv2Sigma60"),
    (524, 80.0, "CoronaryPseudoROIv2Sigma80"),
]
NOGAUSS_TARGET = (525, "CoronaryPseudoROIv2NoGaussian")

SITK_SPACING = (1.0, 1.0, 1.0)
SITK_ORIGIN = (0.0, 0.0, 0.0)
SITK_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def transform_one(args):
    src_ch1, dst_ch1, power = args
    if os.path.exists(dst_ch1):
        return
    img = sitk.ReadImage(src_ch1)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.power(arr, np.float32(power), dtype=np.float32)
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(SITK_SPACING); out.SetOrigin(SITK_ORIGIN); out.SetDirection(SITK_DIRECTION)
    sitk.WriteImage(out, dst_ch1)


def hardlink_resolve(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    real = src.resolve()
    os.link(real, dst)


def build_sigma(dst_id: int, sigma: float, name: str, workers: int) -> None:
    dst = RAW / f"Dataset{dst_id}_{name}"
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    power = (BASE_SIGMA / sigma) ** 2

    ch1_tasks: list[tuple[str, str, float]] = []
    n_hl = 0
    for src in sorted((SRC_DS / "imagesTr").iterdir()):
        nm = src.name
        if nm.endswith("_0000.nii.gz"):
            hardlink_resolve(src, dst / "imagesTr" / nm); n_hl += 1
        elif nm.endswith("_0001.nii.gz"):
            ch1_tasks.append((str(src), str(dst / "imagesTr" / nm), power))

    n_lbl = 0
    for src in sorted((SRC_DS / "labelsTr").iterdir()):
        hardlink_resolve(src, dst / "labelsTr" / src.name); n_lbl += 1

    print(f"  [{dst_id}] hardlinked ch0={n_hl}, labels={n_lbl}; transform ch1={len(ch1_tasks)} (power={power:.4f})")
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(transform_one, ch1_tasks, chunksize=64):
            n_done += 1
            if n_done % 5000 == 0:
                print(f"    {n_done}/{len(ch1_tasks)}")

    with open(SRC_DS / "dataset.json") as f:
        meta = json.load(f)
    meta["name"] = name
    meta["base_sigma"] = sigma
    meta["release"] = "2026-05-19"
    meta["description"] = (
        f"Sigma ablation of Dataset515: identical cases, labels and channel 0 "
        f"(angiography); channel 1 (roi_gaussian) regenerated at sigma={sigma:.0f}. "
        f"Built from Dataset515 ch1 via g_new = g_old^(40^2/{sigma:.0f}^2)."
    )
    meta["ablation_note"] = (
        f"Sigma ablation of Dataset515. ch1 recomputed at sigma={sigma:.0f}; "
        f"ch0 and labels are hardlinks to original Dataset515."
    )
    with open(dst / "dataset.json", "w") as f:
        json.dump(meta, f, indent=2)

    src_splits = SRC_DS / "splits_final.json"
    if src_splits.exists():
        shutil.copy2(src_splits, dst / "splits_final.json")
    print(f"  [{dst_id}] done -> {dst}")


def build_nogauss(dst_id: int, name: str) -> None:
    dst = RAW / f"Dataset{dst_id}_{name}"
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    n_hl = 0
    for src in sorted((SRC_DS / "imagesTr").iterdir()):
        nm = src.name
        if nm.endswith("_0000.nii.gz"):
            hardlink_resolve(src, dst / "imagesTr" / nm); n_hl += 1
        # skip ch1 entirely

    n_lbl = 0
    for src in sorted((SRC_DS / "labelsTr").iterdir()):
        hardlink_resolve(src, dst / "labelsTr" / src.name); n_lbl += 1

    print(f"  [{dst_id}] no-gauss: hardlinked ch0={n_hl}, labels={n_lbl} (no ch1)")

    with open(SRC_DS / "dataset.json") as f:
        meta = json.load(f)
    meta["channel_names"] = {"0": "angiography"}
    meta["name"] = name
    meta.pop("base_sigma", None)
    meta["release"] = "2026-05-19"
    meta["description"] = (
        "Single-channel ablation of Dataset515: angiography only, no Gaussian "
        "attention. Same cases/labels as Dataset515 for direct comparison. "
        "Baseline for measuring the contribution of the Gaussian channel."
    )
    meta["ablation_note"] = (
        "Drop-channel ablation of Dataset515. Pair with 515/521-524 to measure "
        "the value of the Gaussian channel and its sigma sweep."
    )
    with open(dst / "dataset.json", "w") as f:
        json.dump(meta, f, indent=2)

    src_splits = SRC_DS / "splits_final.json"
    if src_splits.exists():
        shutil.copy2(src_splits, dst / "splits_final.json")
    print(f"  [{dst_id}] done -> {dst}")


def main() -> None:
    workers = max(1, (os.cpu_count() or 4) - 2)
    if not (SRC_DS / "imagesTr").exists():
        raise SystemExit(f"Dataset515 not built yet at {SRC_DS}")
    for dst_id, sigma, name in SIGMA_TARGETS:
        print(f"\n=== Dataset{dst_id} sigma={sigma} ===")
        build_sigma(dst_id, sigma, name, workers)
    print(f"\n=== Dataset{NOGAUSS_TARGET[0]} no-gaussian ===")
    build_nogauss(*NOGAUSS_TARGET)


if __name__ == "__main__":
    main()
