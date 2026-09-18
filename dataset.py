"""
Dataset for the MAMA-MIA 100-case preprocessed cache (Section 6/12).

Reads case IDs from split_seed.json and loads the corresponding cached
.pt tensor pairs (written by preprocess_mama_mia.py). Training-only
augmentation via MONAI dictionary transforms: flips, small rotations/
translations, and MILD intensity jitter only -- no aggressive brightness/
contrast jitter and no CLAHE, since contrast enhancement is the diagnostic
signal in DCE-MRI (Section 12).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def build_train_transforms():
    return Compose(
        [
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
            RandAffined(
                keys=["image", "mask"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),   # radians, small rotations only
                translate_range=(5, 5, 5),       # voxels
                padding_mode="border",
            ),
            # Mild intensity jitter only -- enhancement kinetics are the
            # diagnostic signal, aggressive jitter risks erasing it.
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
        ]
    )


class MamaMiaDataset(Dataset):
    def __init__(self, split_json_path, cache_dir, split="train"):
        assert split in ("train", "val", "test")
        with open(split_json_path, "r") as f:
            split_data = json.load(f)
        self.case_ids = split_data["split"][split]
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.transforms = build_train_transforms() if split == "train" else None

        missing = [cid for cid in self.case_ids if not (self.cache_dir / f"{cid}.pt").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) listed in the split are missing from "
                f"{self.cache_dir}: {missing[:5]}..."
            )

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        data = torch.load(self.cache_dir / f"{case_id}.pt")
        item = {"image": data["image"], "mask": data["mask"]}

        if self.transforms is not None:
            item = self.transforms(item)

        return {
            "image": item["image"].as_tensor() if hasattr(item["image"], "as_tensor") else item["image"],
            "mask": item["mask"].as_tensor() if hasattr(item["mask"], "as_tensor") else item["mask"],
            "case_id": case_id,
            "cohort": data["cohort"],
        }