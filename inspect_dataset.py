import os
import json
import numpy as np
import pandas as pd
import nibabel as nib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\aipmu\Desktop\MAMA-MIA\MAMA_MIA_100"

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_CSV = os.path.join(BASE_DIR, "dataset_inspection_report.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "dataset_inspection_summary.json")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_cohort(case_id):
    if case_id.startswith("ISPY1"):
        return "ISPY1"
    elif case_id.startswith("ISPY2"):
        return "ISPY2"
    elif case_id.startswith("NACT"):
        return "NACT"
    elif case_id.startswith("DUKE"):
        return "Duke"
    return "Unknown"


def get_case_id(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


# ============================================================
# FIND FILES
# ============================================================

image_files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz")
)

mask_files = sorted(
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".nii.gz")
)

print("=" * 75)
print("MAMA-MIA 100-CASE DATASET INSPECTION")
print("=" * 75)

print(f"\nImages found: {len(image_files)}")
print(f"Masks found : {len(mask_files)}")


# ============================================================
# INSPECT EACH CASE
# ============================================================

records = []

for i, image_file in enumerate(image_files, start=1):

    case_id = get_case_id(image_file)
    mask_file = image_file

    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    print(f"\n[{i}/{len(image_files)}] {case_id}")

    try:
        # ----------------------------------------------------
        # LOAD NIFTI
        # ----------------------------------------------------

        img = nib.load(image_path)
        mask = nib.load(mask_path)

        # Shape
        image_shape = img.shape
        mask_shape = mask.shape

        # Voxel spacing
        image_spacing = img.header.get_zooms()[:3]
        mask_spacing = mask.header.get_zooms()[:3]

        # Orientation
        image_orientation = "".join(nib.aff2axcodes(img.affine))
        mask_orientation = "".join(nib.aff2axcodes(mask.affine))

        # Affine comparison
        affine_match = np.allclose(img.affine, mask.affine, atol=1e-4)

        # ----------------------------------------------------
        # IMAGE DATA
        # ----------------------------------------------------

        image_data = np.asarray(img.dataobj, dtype=np.float32)

        finite_values = image_data[np.isfinite(image_data)]

        if finite_values.size > 0:

            image_min = float(np.min(finite_values))
            image_max = float(np.max(finite_values))
            image_mean = float(np.mean(finite_values))
            image_std = float(np.std(finite_values))

            p01 = float(np.percentile(finite_values, 1))
            p05 = float(np.percentile(finite_values, 5))
            p50 = float(np.percentile(finite_values, 50))
            p95 = float(np.percentile(finite_values, 95))
            p99 = float(np.percentile(finite_values, 99))

            nonzero = finite_values[finite_values != 0]

            if nonzero.size > 0:
                nonzero_mean = float(np.mean(nonzero))
                nonzero_std = float(np.std(nonzero))
            else:
                nonzero_mean = 0.0
                nonzero_std = 0.0

        else:
            image_min = image_max = image_mean = image_std = np.nan
            p01 = p05 = p50 = p95 = p99 = np.nan
            nonzero_mean = nonzero_std = np.nan

        # ----------------------------------------------------
        # MASK DATA
        # ----------------------------------------------------

        mask_data = np.asarray(mask.dataobj)

        unique_mask_values = np.unique(mask_data)

        tumor_voxels = int(np.count_nonzero(mask_data))

        if tumor_voxels > 0:

            tumor_coords = np.argwhere(mask_data > 0)

            bbox_min = tumor_coords.min(axis=0)
            bbox_max = tumor_coords.max(axis=0)

            bbox_size = bbox_max - bbox_min + 1

            bbox_min_str = ",".join(map(str, bbox_min))
            bbox_max_str = ",".join(map(str, bbox_max))
            bbox_size_str = ",".join(map(str, bbox_size))

        else:
            bbox_min_str = ""
            bbox_max_str = ""
            bbox_size_str = ""

        # ----------------------------------------------------
        # PRINT QUICK INFO
        # ----------------------------------------------------

        print(f"  Shape       : {image_shape}")
        print(f"  Spacing     : {tuple(round(x, 4) for x in image_spacing)}")
        print(f"  Orientation : {image_orientation}")
        print(f"  Mask values : {unique_mask_values.tolist()}")
        print(f"  Tumor voxels: {tumor_voxels}")

        # ----------------------------------------------------
        # SAVE RECORD
        # ----------------------------------------------------

        records.append({
            "case_id": case_id,
            "cohort": get_cohort(case_id),

            "image_shape": str(image_shape),
            "mask_shape": str(mask_shape),

            "spacing_x": float(image_spacing[0]),
            "spacing_y": float(image_spacing[1]),
            "spacing_z": float(image_spacing[2]),

            "image_orientation": image_orientation,
            "mask_orientation": mask_orientation,

            "affine_match": bool(affine_match),
            "shape_match": bool(image_shape == mask_shape),

            "image_min": image_min,
            "image_max": image_max,
            "image_mean": image_mean,
            "image_std": image_std,

            "p01": p01,
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "p99": p99,

            "nonzero_mean": nonzero_mean,
            "nonzero_std": nonzero_std,

            "mask_unique_values": str(unique_mask_values.tolist()),
            "tumor_voxels": tumor_voxels,

            "tumor_bbox_min": bbox_min_str,
            "tumor_bbox_max": bbox_max_str,
            "tumor_bbox_size": bbox_size_str,
        })

    except Exception as e:

        print(f"  ERROR: {e}")

        records.append({
            "case_id": case_id,
            "cohort": get_cohort(case_id),
            "error": str(e)
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "total_cases": int(len(df)),
    "cohort_counts": df["cohort"].value_counts().to_dict(),

    "unique_image_shapes": df["image_shape"].value_counts().to_dict(),

    "unique_spacings": (
        df[
            ["spacing_x", "spacing_y", "spacing_z"]
        ]
        .round(4)
        .astype(str)
        .agg(" × ".join, axis=1)
        .value_counts()
        .to_dict()
    ),

    "image_orientations": df["image_orientation"].value_counts().to_dict(),

    "mask_orientations": df["mask_orientation"].value_counts().to_dict(),

    "shape_matches": int(df["shape_match"].sum()),

    "affine_matches": int(df["affine_match"].sum()),

    "empty_masks": int((df["tumor_voxels"] == 0).sum()),
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("DATASET INSPECTION SUMMARY")
print("=" * 75)

print(f"\nTotal cases: {len(df)}")

print("\nCohorts:")
print(df["cohort"].value_counts().sort_index())

print("\nImage shapes:")
print(df["image_shape"].value_counts())

print("\nOrientations:")
print(df["image_orientation"].value_counts())

print("\nShape matches:")
print(f"{df['shape_match'].sum()} / {len(df)}")

print("\nAffine matches:")
print(f"{df['affine_match'].sum()} / {len(df)}")

print("\nEmpty masks:")
print(f"{(df['tumor_voxels'] == 0).sum()}")

print("\nReports saved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("\n" + "=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)