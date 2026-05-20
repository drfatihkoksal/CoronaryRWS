"""Build ARCADE-test-only inference input folders for the v2 sigma ablation.

For each σ in {20, 30, 40, 60, 80} and the no-gaussian variant, create a
folder under predictions/Dataset510_test_inputs/sigma{XX}/ containing
nnU-Net-formatted inputs derived from Dataset510 case files but filtered to
only those whose case_id starts with 'arcade_test_'.

ch0 is hardlinked from Dataset510 (same angiography).
ch1 is power-transformed (or omitted for no-gauss):
    g_new = g_old ^ (40^2 / σ^2)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SRC = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset510_CoronaryARCADE/imagesTr")
DST_ROOT = Path("/media/fatih/1tb/training_data/predictions/Dataset510_test_inputs")
BASE_SIGMA = 40.0
SIGMAS = [20.0, 30.0, 40.0, 60.0, 80.0]

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


def collect_test_pairs():
    """Return list of (ch0_path, ch1_path, case_id) for arcade_test_* cases."""
    pairs = []
    seen = set()
    for f in sorted(SRC.glob("arcade_test_*_0000.nii.gz")):
        case_id = f.name.replace("_0000.nii.gz", "")
        if case_id in seen: continue
        ch1 = SRC / f"{case_id}_0001.nii.gz"
        if not ch1.exists(): continue
        pairs.append((f, ch1, case_id))
        seen.add(case_id)
    return pairs


def build_sigma(sigma: float, pairs, workers: int) -> None:
    name = f"sigma{int(sigma)}"
    dst = DST_ROOT / name
    dst.mkdir(parents=True, exist_ok=True)
    power = (BASE_SIGMA / sigma) ** 2

    for ch0, _, cid in pairs:
        hardlink_resolve(ch0, dst / f"{cid}_0000.nii.gz")

    ch1_tasks = [(str(c1), str(dst / f"{cid}_0001.nii.gz"), power) for _, c1, cid in pairs]
    print(f"  sigma={sigma:.0f}: hardlinked {len(pairs)} ch0; transform ch1 (power={power:.4f})")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(transform_one, ch1_tasks, chunksize=64):
            pass


def build_nogauss(pairs) -> None:
    dst = DST_ROOT / "nogauss"
    dst.mkdir(parents=True, exist_ok=True)
    for ch0, _, cid in pairs:
        hardlink_resolve(ch0, dst / f"{cid}_0000.nii.gz")
    print(f"  no-gauss: hardlinked {len(pairs)} ch0 (no ch1)")


def main():
    workers = max(1, (os.cpu_count() or 4) - 2)
    pairs = collect_test_pairs()
    print(f"arcade_test_* cases: {len(pairs)}")
    if not pairs:
        raise SystemExit("no test cases found in Dataset510 imagesTr")
    for s in SIGMAS:
        build_sigma(s, pairs, workers)
    build_nogauss(pairs)
    print("\ndone")


if __name__ == "__main__":
    main()
