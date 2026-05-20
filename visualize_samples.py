"""Visualize Dataset515 ROI crops: ch0 | ch1 | label | overlay."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

DST = Path("/media/fatih/1tb/training_data/nnunet_raw/Dataset515_CoronaryPseudoROI_v2")
OUT = Path("/media/fatih/1tb/training_data/viz_samples")
OUT.mkdir(exist_ok=True)

CASES = [
    "r20a2ef1a_XRayAngiography210376114_f0031_c24",
    "r482ebe91_XRayAngiography769469KEN_f0061_c05",
    "r5da7673c_XRayAngiography886552TUZ_f0040_c06",
    "r9dee75f9_XRayAngiography13881242O_f0036_c01",
    "rc67438e7_XRayAngiography1346401HA_f0023_c07",
    "rd7fb2a11_XRayAngiography00673100D_f0042_c07",
]

def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).squeeze().astype(np.float32)

def to_u8(arr, vmin=None, vmax=None):
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    return np.clip((arr - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

for cid in CASES:
    ch0 = load(DST / "imagesTr" / f"{cid}_0000.nii.gz")
    ch1 = load(DST / "imagesTr" / f"{cid}_0001.nii.gz")
    lbl = load(DST / "labelsTr" / f"{cid}.nii.gz")

    img0 = Image.fromarray(to_u8(ch0)).convert("RGB")
    img1 = Image.fromarray(to_u8(ch1, 0, 1)).convert("RGB")
    img_lbl = Image.fromarray(((lbl > 0).astype(np.uint8) * 255)).convert("RGB")

    overlay = img0.copy().convert("RGBA")
    mask_rgba = np.zeros((*lbl.shape, 4), dtype=np.uint8)
    mask_rgba[lbl > 0] = [255, 50, 50, 160]
    overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba)).convert("RGB")

    H, W = ch0.shape
    strip = Image.new("RGB", (W * 4 + 9, H + 20), (30, 30, 30))
    cls = cid.split("_c")[-1]
    for i, (panel, txt) in enumerate(zip(
        [img0, img1, img_lbl, overlay],
        ["angio (ch0)", "gaussian (ch1)", "label", "overlay"]
    )):
        x = i * (W + 3)
        strip.paste(panel, (x, 20))
        ImageDraw.Draw(strip).text((x + 2, 2), f"{txt} | c{cls}", fill=(220, 220, 100))

    strip.save(OUT / f"{cid}.png")
    print(f"saved {cid[-20:]}")
