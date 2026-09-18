import os
import re
import json
import nibabel as nib
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "MAMA_MIA_100"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

EXPECTED_TOTAL = 100
EXPECTED_PER_COHORT = 25

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_case_id(filename):
    """
    Extract case ID from filenames such as:
        ISPY1_1082.nii.gz
        ISPY2_1234.nii.gz
        NACT_001.nii.gz
        DUKE_001.nii.gz
    """

    name = filename.replace(".nii.gz", "").replace(".nii", "")

    # Keep the complete cohort + identifier
    match = re.match(r"^(ISPY1|ISPY2|NACT|DUKE)_(.+)$", name)

    if match:
        return f"{match.group(1)}_{match.group(2)}"

    return None


def get_cohort(case_id):
    if case_id.startswith("ISPY1_"):
        return "ISPY1"
    elif case_id.startswith("ISPY2_"):
        return "ISPY2"
    elif case_id.startswith("NACT_"):
        return "NACT"
    elif case_id.startswith("DUKE_"):
        return "Duke"
    else:
        return "Unknown"


def load_nifti(path):
    """
    Load NIfTI safely and return image object + data.
    """
    img = nib.load(path)
    data = img.get_fdata()
    return img, data


# ============================================================
# 1. CHECK DIRECTORIES
# ============================================================

print("=" * 70)
print("MAMA-MIA 100-CASE DATASET VERIFICATION")
print("=" * 70)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found:\n{os.path.abspath(DATASET_DIR)}"
    )

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Images directory not found:\n{os.path.abspath(IMAGE_DIR)}"
    )

if not os.path.isdir(MASK_DIR):
    raise FileNotFoundError(
        f"Masks directory not found:\n{os.path.abspath(MASK_DIR)}"
    )

print("\nDataset directory:")
print(os.path.abspath(DATASET_DIR))


# ============================================================
# 2. FIND NIFTI FILES
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz") or f.endswith(".nii")
])

mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".nii.gz") or f.endswith(".nii")
])

print("\n" + "-" * 70)
print("FILE COUNTS")
print("-" * 70)

print(f"Image files : {len(image_files)}")
print(f"Mask files  : {len(mask_files)}")


# ============================================================
# 3. CHECK EXACTLY 100
# ============================================================

if len(image_files) == EXPECTED_TOTAL:
    print("✓ Exactly 100 MRI image files found.")
else:
    print(f"✗ Expected 100 images but found {len(image_files)}.")

if len(mask_files) == EXPECTED_TOTAL:
    print("✓ Exactly 100 mask files found.")
else:
    print(f"✗ Expected 100 masks but found {len(mask_files)}.")


# ============================================================
# 4. EXTRACT CASE IDs
# ============================================================

image_cases = {}

for filename in image_files:
    case_id = get_case_id(filename)

    if case_id is None:
        print(f"⚠ Could not identify case ID from image: {filename}")
    else:
        image_cases[case_id] = filename


mask_cases = {}

for filename in mask_files:
    case_id = get_case_id(filename)

    if case_id is None:
        print(f"⚠ Could not identify case ID from mask: {filename}")
    else:
        mask_cases[case_id] = filename


image_ids = set(image_cases.keys())
mask_ids = set(mask_cases.keys())


# ============================================================
# 5. CHECK IMAGE-MASK MATCHING
# ============================================================

print("\n" + "-" * 70)
print("IMAGE ↔ MASK MATCHING")
print("-" * 70)

missing_masks = sorted(image_ids - mask_ids)
missing_images = sorted(mask_ids - image_ids)

if not missing_masks:
    print("✓ Every MRI image has a corresponding mask.")
else:
    print("✗ Images without masks:")
    for case in missing_masks:
        print("  ", case)

if not missing_images:
    print("✓ Every mask has a corresponding MRI image.")
else:
    print("✗ Masks without images:")
    for case in missing_images:
        print("  ", case)

paired_cases = sorted(image_ids & mask_ids)

print(f"\nPaired cases: {len(paired_cases)}")

if len(paired_cases) == 100:
    print("✓ Exactly 100 complete image-mask pairs.")
else:
    print(f"✗ Expected 100 pairs but found {len(paired_cases)}.")


# ============================================================
# 6. CHECK COHORT DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("COHORT DISTRIBUTION")
print("-" * 70)

cohort_counts = {
    "ISPY1": 0,
    "ISPY2": 0,
    "NACT": 0,
    "Duke": 0,
    "Unknown": 0
}

for case_id in paired_cases:
    cohort = get_cohort(case_id)
    cohort_counts[cohort] += 1

for cohort, count in cohort_counts.items():
    if cohort == "Unknown":
        continue

    expected = EXPECTED_PER_COHORT

    if count == expected:
        status = "✓"
    else:
        status = "✗"

    print(f"{status} {cohort}: {count} cases (expected {expected})")


# ============================================================
# 7. CHECK DUPLICATE CASE IDs
# ============================================================

print("\n" + "-" * 70)
print("DUPLICATE CHECK")
print("-" * 70)

if len(image_cases) == len(image_files):
    print("✓ No duplicate image case IDs detected.")
else:
    print("✗ Duplicate image case IDs may exist.")

if len(mask_cases) == len(mask_files):
    print("✓ No duplicate mask case IDs detected.")
else:
    print("✗ Duplicate mask case IDs may exist.")


# ============================================================
# 8. LOAD AND VERIFY EVERY MRI + MASK
# ============================================================

print("\n" + "-" * 70)
print("NIFTI / SHAPE / MASK VERIFICATION")
print("-" * 70)

results = []

invalid_images = []
invalid_masks = []
shape_mismatches = []
empty_masks = []

for i, case_id in enumerate(paired_cases, start=1):

    image_path = os.path.join(
        IMAGE_DIR,
        image_cases[case_id]
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_cases[case_id]
    )

    row = {
        "case_id": case_id,
        "cohort": get_cohort(case_id),
        "image_file": image_cases[case_id],
        "mask_file": mask_cases[case_id],
        "image_valid": False,
        "mask_valid": False,
        "image_shape": None,
        "mask_shape": None,
        "shape_match": False,
        "mask_nonempty": False,
        "mask_voxels": 0,
        "status": "ERROR"
    }

    # ----------------------------
    # Load MRI
    # ----------------------------

    try:
        image_obj, image_data = load_nifti(image_path)

        row["image_valid"] = True
        row["image_shape"] = tuple(image_data.shape)

    except Exception as e:
        invalid_images.append((case_id, str(e)))
        results.append(row)
        continue

    # ----------------------------
    # Load mask
    # ----------------------------

    try:
        mask_obj, mask_data = load_nifti(mask_path)

        row["mask_valid"] = True
        row["mask_shape"] = tuple(mask_data.shape)

    except Exception as e:
        invalid_masks.append((case_id, str(e)))
        results.append(row)
        continue

    # ----------------------------
    # Shape check
    # ----------------------------

    if image_data.shape == mask_data.shape:

        row["shape_match"] = True

    else:

        shape_mismatches.append(
            (
                case_id,
                image_data.shape,
                mask_data.shape
            )
        )

    # ----------------------------
    # Mask non-empty check
    # ----------------------------

    tumor_voxels = int(np.count_nonzero(mask_data))

    row["mask_voxels"] = tumor_voxels

    if tumor_voxels > 0:
        row["mask_nonempty"] = True
    else:
        empty_masks.append(case_id)

    # ----------------------------
    # Overall status
    # ----------------------------

    if (
        row["image_valid"]
        and row["mask_valid"]
        and row["shape_match"]
        and row["mask_nonempty"]
    ):
        row["status"] = "OK"

    results.append(row)

    # Progress
    if i % 10 == 0 or i == len(paired_cases):
        print(f"Checked {i}/{len(paired_cases)} cases...")


# ============================================================
# 9. SUMMARY
# ============================================================

df = pd.DataFrame(results)

valid_count = int((df["status"] == "OK").sum())

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"Total paired cases       : {len(paired_cases)}")
print(f"Valid complete cases     : {valid_count}")
print(f"Invalid MRI files        : {len(invalid_images)}")
print(f"Invalid mask files       : {len(invalid_masks)}")
print(f"Shape mismatches         : {len(shape_mismatches)}")
print(f"Empty tumor masks        : {len(empty_masks)}")


# ============================================================
# 10. PRINT PROBLEMS
# ============================================================

if invalid_images:

    print("\n✗ INVALID MRI FILES")
    for case, error in invalid_images:
        print(f"{case}: {error}")


if invalid_masks:

    print("\n✗ INVALID MASK FILES")
    for case, error in invalid_masks:
        print(f"{case}: {error}")


if shape_mismatches:

    print("\n✗ SHAPE MISMATCHES")

    for case, image_shape, mask_shape in shape_mismatches:

        print(
            f"{case}: "
            f"image={image_shape}, "
            f"mask={mask_shape}"
        )


if empty_masks:

    print("\n✗ EMPTY MASKS")

    for case in empty_masks:
        print(case)


# ============================================================
# 11. SHOW COHORT TABLE
# ============================================================

print("\n" + "-" * 70)
print("FINAL COHORT TABLE")
print("-" * 70)

cohort_table = (
    df.groupby("cohort")
    .agg(
        cases=("case_id", "count"),
        valid=("status", lambda x: (x == "OK").sum())
    )
    .reset_index()
)

print(cohort_table.to_string(index=False))


# ============================================================
# 12. SAVE VERIFICATION REPORT
# ============================================================

report_path = os.path.join(
    DATASET_DIR,
    "dataset_verification_report.csv"
)

df.to_csv(report_path, index=False)

print("\n✓ Detailed report saved to:")
print(os.path.abspath(report_path))


# ============================================================
# 13. SAVE CASE-ID LIST
# ============================================================

case_list_path = os.path.join(
    DATASET_DIR,
    "selected_case_ids.json"
)

selected_case_data = {
    "total_cases": len(paired_cases),
    "cohort_counts": cohort_counts,
    "case_ids": paired_cases
}

with open(case_list_path, "w") as f:
    json.dump(selected_case_data, f, indent=4)

print("✓ Case-ID list saved to:")
print(os.path.abspath(case_list_path))


# ============================================================
# 14. FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET STATUS")
print("=" * 70)

if (
    len(image_files) == 100
    and len(mask_files) == 100
    and len(paired_cases) == 100
    and valid_count == 100
    and all(cohort_counts[c] == 25 for c in ["ISPY1", "ISPY2", "NACT", "Duke"])
):

    print("✓ DATASET PASSED BASIC VERIFICATION")
    print()
    print("100 valid image-mask pairs are ready for preprocessing.")
    print("Next step: inspect MRI/mask orientation, spacing and intensity")
    print("before implementing the PyTorch 3D dataset loader.")

else:

    print("✗ DATASET DID NOT PASS VERIFICATION")
    print()
    print("Do NOT start model training yet.")
    print("Inspect the errors reported above.")