"""
MAMA-MIA 100-case preprocessing pipeline
=========================================

Implements Section 5 of the project proposal:
  LoadImaged -> Orientationd(RAS) -> Spacingd (isotropic) ->
  CropForegroundd (IMAGE-derived, never mask-derived) ->
  [ANALYZE crop-size distribution] ->
  ResizeWithPadOrCropd (fixed cubic target) -> NormalizeIntensityd ->
  verify binary mask + nonzero tumor voxels -> save .pt cache

Two-step, informed workflow (do not skip step 1):

  Step 1 - ANALYZE (no target size needed, nothing is saved):
      python preprocess_mama_mia.py --data-dir "C:\\path\\to\\MAMA_MIA_100" --analyze-only

  Step 2 - PROCESS (pick a cubic target size based on Step 1's printed
  percentiles, e.g. 128 or 96, then run the full pipeline and save the cache):
      python preprocess_mama_mia.py --data-dir "C:\\path\\to\\MAMA_MIA_100" --target-size 128

Requires: monai, nibabel, torch, pandas, tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
)


def otsu_threshold(x: np.ndarray, n_bins: int = 256) -> float:
    """Simple Otsu's-method threshold, computed per-case on the raw intensity
    array, to separate breast tissue from background air/noise. MRI background
    is not literally zero (scanner noise sits just above 0), so a fixed
    `x > 0` check keeps almost the entire field of view as "foreground" --
    this is what the analysis run surfaced (100/100 cases exceeding 160^3,
    tracking the raw FOV size almost exactly)."""
    vals = x[x > 0]  # ignore true zero padding, if any
    if vals.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(vals, bins=n_bins)
    hist = hist.astype(np.float64)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    total = weight1[-1]
    if total == 0:
        return 0.0

    mean1 = np.cumsum(hist * bin_mids) / np.clip(weight1, 1e-12, None)
    mean2 = (np.cumsum((hist * bin_mids)[::-1])[::-1]) / np.clip(weight2, 1e-12, None)

    between_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(between_class_var))
    return float(bin_mids[idx])


def _otsu_select_fn(x):
    """select_fn for CropForegroundd: computes an Otsu threshold on this
    case's own intensity array and returns a boolean foreground mask.
    Replaces the old `x > 0` check, which barely trims MRI background noise."""
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    thr = otsu_threshold(arr)
    return x > thr


def discover_pairs(data_dir: Path):
    """Match images/<case>.nii.gz to masks/<case>.nii.gz by filename stem."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        sys.exit(f"Expected 'images/' and 'masks/' subfolders under {data_dir}")

    def stem(p: Path) -> str:
        # strips .nii.gz (two suffixes) or .nii
        name = p.name
        if name.endswith(".nii.gz"):
            return name[: -len(".nii.gz")]
        return p.stem

    images = {stem(p): p for p in images_dir.glob("*.nii*")}
    masks = {stem(p): p for p in masks_dir.glob("*.nii*")}
    common = sorted(set(images) & set(masks))
    missing_mask = sorted(set(images) - set(masks))
    missing_image = sorted(set(masks) - set(images))
    if missing_mask:
        print(f"WARNING: {len(missing_mask)} images have no matching mask, skipped: {missing_mask[:5]}...")
    if missing_image:
        print(f"WARNING: {len(missing_image)} masks have no matching image, skipped: {missing_image[:5]}...")

    pairs = [(cid, images[cid], masks[cid]) for cid in common]
    return pairs


def cohort_of(case_id: str) -> str:
    for name in ("ISPY1", "ISPY2", "NACT", "DUKE", "Duke"):
        if case_id.upper().startswith(name.upper()):
            return "Duke" if name.upper() == "DUKE" else name
    return "UNKNOWN"


def build_pre_crop_transforms(spacing_mm: float):
    """Everything up to and including the image-derived foreground crop."""
    return Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(spacing_mm, spacing_mm, spacing_mm),
                mode=("bilinear", "nearest"),
            ),
            # IMAGE-derived crop only -- never derive the crop from the mask.
            # A mask-derived crop is untestable at inference time (you don't
            # know where the tumor is -- that's the task) and would create a
            # train/test distribution mismatch.
            CropForegroundd(
                keys=["image", "mask"],
                source_key="image",
                select_fn=_otsu_select_fn,
                margin=10,
            ),
        ]
    )


def build_post_crop_transforms(target_size: int):
    """Fixed cubic size + intensity normalization, applied after analysis."""
    size = (target_size, target_size, target_size)
    return Compose(
        [
            # Pads if the crop is smaller than target, center-crops if larger.
            # This preserves isotropic physical scale (no stretching/distortion),
            # unlike a plain Resize which would distort aspect ratio.
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=size),
            # Percentile clipping for outlier robustness, then z-score-like
            # scaling into [0, 1]; swap for NormalizeIntensityd if you prefer
            # pure z-score instead of range scaling.
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )


def run_analysis(pairs, spacing_mm, out_dir: Path):
    """Step 1: resample + crop every case, report the crop-size distribution
    so a cubic target size can be chosen from real data instead of guessed."""
    pre_crop = build_pre_crop_transforms(spacing_mm)
    rows = []

    for case_id, img_path, mask_path in tqdm(pairs, desc="Analyzing"):
        data = {"image": str(img_path), "mask": str(mask_path)}
        try:
            out = pre_crop(data)
        except Exception as e:
            rows.append({"case_id": case_id, "cohort": cohort_of(case_id), "error": str(e)})
            continue

        img_shape = tuple(out["image"].shape[1:])  # drop channel dim
        mask_arr = out["mask"].numpy()
        tumor_voxels = int((mask_arr > 0).sum())
        rows.append(
            {
                "case_id": case_id,
                "cohort": cohort_of(case_id),
                "cropped_shape_x": img_shape[0],
                "cropped_shape_y": img_shape[1],
                "cropped_shape_z": img_shape[2],
                "max_dim": max(img_shape),
                "tumor_voxels_after_crop": tumor_voxels,
                "error": "",
            }
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "crop_size_analysis.csv"
    df.to_csv(report_path, index=False)

    ok = df[df["error"] == ""]
    print("\n" + "=" * 70)
    print("CROP-SIZE ANALYSIS (after isotropic resample + image-derived crop)")
    print("=" * 70)
    if len(ok) == 0:
        print("No cases processed successfully -- check errors in", report_path)
        return

    for axis in ("cropped_shape_x", "cropped_shape_y", "cropped_shape_z"):
        vals = ok[axis]
        print(
            f"{axis:>20}: min={vals.min():4d}  p50={vals.median():6.1f}  "
            f"p90={vals.quantile(0.9):6.1f}  p95={vals.quantile(0.95):6.1f}  max={vals.max():4d}"
        )

    print(f"\n{'max_dim overall':>20}: min={ok['max_dim'].min()}  p90={ok['max_dim'].quantile(0.9):.1f}  "
          f"p95={ok['max_dim'].quantile(0.95):.1f}  max={ok['max_dim'].max()}")

    for candidate in (96, 112, 128, 144, 160):
        n_clipped = int((ok["max_dim"] > candidate).sum())
        print(f"  -> cubic target {candidate:>3}: would CENTER-CROP (possibly clip content) "
              f"in {n_clipped}/{len(ok)} cases")

    n_zero_tumor = int((ok["tumor_voxels_after_crop"] == 0).sum())
    if n_zero_tumor:
        print(f"\nWARNING: {n_zero_tumor} case(s) have ZERO tumor voxels after the crop step "
              f"-- the foreground crop is clipping the tumor out. Inspect these before proceeding:")
        print(ok[ok["tumor_voxels_after_crop"] == 0][["case_id", "cohort"]].to_string(index=False))

    print(f"\nFull per-case report saved to: {report_path}")
    print("\nPick a cubic target size above (ideally one where 0, or very few, cases get "
          "clipped), then re-run with --target-size <N>.")


def run_process(pairs, spacing_mm, target_size, out_dir: Path, small_tumor_voxel_threshold=50):
    """Step 2: full pipeline, saves .pt tensor pairs + a per-case report."""
    pre_crop = build_pre_crop_transforms(spacing_mm)
    post_crop = build_post_crop_transforms(target_size)

    cache_dir = out_dir / f"MAMA_MIA_100_preprocessed_{target_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id, img_path, mask_path in tqdm(pairs, desc=f"Processing (target={target_size}^3)"):
        data = {"image": str(img_path), "mask": str(mask_path)}
        row = {"case_id": case_id, "cohort": cohort_of(case_id)}
        try:
            out = pre_crop(data)
            tumor_before = int((out["mask"].numpy() > 0).sum())
            crop_shape = tuple(out["image"].shape[1:])

            out = post_crop(out)
            mask_arr = out["mask"].numpy()
            uniq = sorted(np.unique(mask_arr).tolist())
            tumor_after = int((mask_arr > 0).sum())

            # --- safety checks, do not silently save a broken case ---
            if uniq not in ([0.0], [1.0], [0.0, 1.0]):
                row["status"] = f"FAIL: mask not binary, unique values={uniq}"
                rows.append(row)
                continue
            if tumor_after == 0:
                row["status"] = "FAIL: tumor mask empty after crop/resize (clipped out)"
                rows.append(row)
                continue
            if tumor_after < small_tumor_voxel_threshold:
                row["status"] = f"WARN: small tumor after resize ({tumor_after} voxels) -- inspect visually"
            else:
                row["status"] = "OK"

            img_t = out["image"].as_tensor().float()  # [1, D, H, W]
            mask_t = out["mask"].as_tensor().float()  # [1, D, H, W]
            torch.save(
                {"image": img_t, "mask": mask_t, "case_id": case_id, "cohort": row["cohort"]},
                cache_dir / f"{case_id}.pt",
            )

            row.update(
                {
                    "cropped_shape": crop_shape,
                    "tumor_voxels_before_resize": tumor_before,
                    "tumor_voxels_after_resize": tumor_after,
                    "pct_tumor_voxels_retained": round(100.0 * tumor_after / max(tumor_before, 1), 1),
                }
            )
        except Exception as e:
            row["status"] = f"ERROR: {e}"
        rows.append(row)

    df = pd.DataFrame(rows)
    report_path = out_dir / f"preprocessing_report_{target_size}.csv"
    df.to_csv(report_path, index=False)

    n_ok = int((df["status"] == "OK").sum())
    n_warn = int(df["status"].astype(str).str.startswith("WARN").sum())
    n_fail = int(df["status"].astype(str).str.startswith(("FAIL", "ERROR")).sum())

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Target cubic size : {target_size}^3")
    print(f"Total cases       : {len(df)}")
    print(f"OK                : {n_ok}")
    print(f"WARN (small tumor): {n_warn}")
    print(f"FAIL/ERROR        : {n_fail}")
    print(f"Cache directory   : {cache_dir}")
    print(f"Full report       : {report_path}")

    if n_fail:
        print("\nCases needing attention (FAIL/ERROR):")
        print(df[df["status"].astype(str).str.startswith(("FAIL", "ERROR"))][["case_id", "cohort", "status"]]
              .to_string(index=False))
    if n_warn:
        print("\nCases with small tumors post-resize (WARN) -- spot-check these visually:")
        print(df[df["status"].astype(str).str.startswith("WARN")][["case_id", "cohort", "status"]]
              .to_string(index=False))

    # Also write a small JSON summary for quick programmatic checks later.
    summary = {
        "target_size": target_size,
        "spacing_mm": spacing_mm,
        "total_cases": len(df),
        "ok": n_ok,
        "warn": n_warn,
        "fail": n_fail,
        "cache_dir": str(cache_dir),
    }
    with open(out_dir / f"preprocessing_summary_{target_size}.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MAMA-MIA 100-case preprocessing pipeline")
    parser.add_argument("--data-dir", required=True, type=Path,
                         help="Path to MAMA_MIA_100 folder (must contain images/ and masks/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Where to write reports and the preprocessed cache "
                              "(default: <data-dir>/preprocessing_output)")
    parser.add_argument("--spacing-mm", type=float, default=1.0,
                         help="Isotropic target spacing in mm (default: 1.0)")
    parser.add_argument("--target-size", type=int, default=None,
                         help="Cubic target size in voxels, e.g. 128. If omitted, "
                              "runs analysis-only and does not save anything.")
    parser.add_argument("--analyze-only", action="store_true",
                         help="Force analysis-only mode even if --target-size is given.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N cases (for a quick smoke test).")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.data_dir / "preprocessing_output")
    pairs = discover_pairs(args.data_dir)
    print(f"Found {len(pairs)} matched image/mask pairs under {args.data_dir}")
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"--limit set: using first {len(pairs)} cases only")

    if args.analyze_only or args.target_size is None:
        run_analysis(pairs, args.spacing_mm, out_dir)
        if args.target_size is None and not args.analyze_only:
            print("\nNo --target-size given, so only the analysis ran. "
                  "Re-run with --target-size <N> once you've picked a size.")
    else:
        run_process(pairs, args.spacing_mm, args.target_size, out_dir)


if __name__ == "__main__":
    main()