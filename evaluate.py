"""
Final evaluation (Section 15-16), scoped to the agreed must-have set:
  Dice, IoU, Precision, Recall, HD95
  training/validation loss curves
  MRI + GT + prediction visualizations for best/median/worst test cases
  dice_distribution.png (histogram across all test cases)

IMPORTANT: --spacing-mm must match whatever isotropic spacing was used in
preprocess_mama_mia.py (you ran --spacing-mm 2.0), or HD95 will be
computed in the wrong physical units silently.

Usage:
    python evaluate.py --checkpoint "...\\training_output\\best_model.pt" ^
                        --cache-dir "...\\MAMA_MIA_100_preprocessed_128" ^
                        --split-json "...\\split_seed.json" ^
                        --history-json "...\\training_output\\training_history.json" ^
                        --out-dir "...\\results" ^
                        --spacing-mm 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import HybridSegModel
from dataset import MamaMiaDataset
from metrics import dice_score, iou_score, precision_recall, hd95_score


def summarize(values, label):
    values = np.array([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {f"{label}_mean": None, f"{label}_median": None,
                f"{label}_iqr_low": None, f"{label}_iqr_high": None, f"{label}_n": 0}
    return {
        f"{label}_mean": float(values.mean()),
        f"{label}_median": float(np.median(values)),
        f"{label}_iqr_low": float(np.percentile(values, 25)),
        f"{label}_iqr_high": float(np.percentile(values, 75)),
        f"{label}_n": len(values),
    }


def run_inference(model, loader, device, spacing_mm):
    model.eval()
    rows = []
    volumes = {}  # case_id -> (mri_np, gt_np, pred_np) for later visualization

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            case_id = batch["case_id"][0]
            cohort = batch["cohort"][0]

            logits = model(x)
            probs = torch.sigmoid(logits)

            dice = dice_score(probs, y).item()
            iou = iou_score(probs, y).item()
            precision, recall = precision_recall(probs, y)
            precision, recall = precision.item(), recall.item()
            hd95 = hd95_score(probs, y, spacing_mm=spacing_mm)

            rows.append({
                "case_id": case_id, "cohort": cohort,
                "dice": dice, "iou": iou,
                "precision": precision, "recall": recall,
                "hd95_mm": hd95,
                "hd95_flag": "undefined (one of pred/GT empty)" if hd95 is None else "",
            })

            volumes[case_id] = (
                x[0, 0].cpu().numpy(),
                y[0, 0].cpu().numpy(),
                (probs[0, 0].cpu().numpy() > 0.5).astype(np.float32),
            )

    return pd.DataFrame(rows), volumes


def plot_loss_curves(history_json_path, out_dir):
    import json
    with open(history_json_path, "r") as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, h["train_loss"], label="Train loss")
    ax.plot(epochs, h["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DiceCELoss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_validation_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, h["val_dice"], color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_title("Validation Dice per Epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "validation_dice.png", dpi=150)
    plt.close(fig)


def plot_dice_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["dice"].values, bins=10, edgecolor="black")
    ax.set_xlabel("Dice score")
    ax.set_ylabel("Number of test cases")
    ax.set_title(f"Test Dice Distribution (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_distribution.png", dpi=150)
    plt.close(fig)


def best_slice_index(gt_np, pred_np):
    """Pick the axial slice with the most GT tumor area; fall back to the
    slice with the most predicted tumor, then the middle slice."""
    gt_areas = gt_np.sum(axis=(1, 2))  # assumes (D,H,W)
    if gt_areas.max() > 0:
        return int(gt_areas.argmax())
    pred_areas = pred_np.sum(axis=(1, 2))
    if pred_areas.max() > 0:
        return int(pred_areas.argmax())
    return gt_np.shape[0] // 2


def save_case_visualization(case_id, dice_val, mri_np, gt_np, pred_np, out_path):
    z = best_slice_index(gt_np, pred_np)
    mri_slice = mri_np[z]
    gt_slice = gt_np[z]
    pred_slice = pred_np[z]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(mri_slice, cmap="gray")
    axes[1].imshow(gt_slice, cmap="Reds", alpha=0.5)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mri_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Blues", alpha=0.5)
    axes[2].set_title("Prediction")
    axes[3].imshow(mri_slice, cmap="gray")
    axes[3].imshow(gt_slice, cmap="Reds", alpha=0.4)
    axes[3].imshow(pred_slice, cmap="Blues", alpha=0.4)
    axes[3].set_title("Overlay (GT=red, Pred=blue)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{case_id} -- Dice = {dice_val:.3f} (axial slice {z})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--history-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, required=True,
                         help="MUST match the --spacing-mm used in preprocess_mama_mia.py")
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    metrics_dir = args.out_dir / "metrics"
    plots_dir = args.out_dir / "plots"
    qual_dir = args.out_dir / "qualitative"
    for d in (metrics_dir, plots_dir, qual_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_qnn = ckpt.get("args", {}).get("use_qnn", True)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_dice={ckpt['val_dice']:.4f}, use_qnn={use_qnn})")

    model = HybridSegModel(
        in_channels=1, base_channels=args.base_channels, out_channels=1, use_qnn=use_qnn
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_ds = MamaMiaDataset(args.split_json, args.cache_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test cases: {len(test_ds)}")

    df, volumes = run_inference(model, test_loader, device, args.spacing_mm)
    df.to_csv(metrics_dir / "case_metrics.csv", index=False)

    n_hd95_undefined = df["hd95_mm"].isna().sum()
    if n_hd95_undefined:
        print(f"\nNOTE: HD95 undefined for {n_hd95_undefined} case(s) "
              f"(prediction or GT was empty) -- excluded from HD95 summary stats, "
              f"flagged in case_metrics.csv, not silently averaged in.")

    # ---- Overall summary ----
    overall = {}
    for col, label in [("dice", "dice"), ("iou", "iou"),
                        ("precision", "precision"), ("recall", "recall"),
                        ("hd95_mm", "hd95_mm")]:
        overall.update(summarize(df[col].tolist(), label))
    pd.DataFrame([overall]).to_csv(metrics_dir / "test_metrics_summary.csv", index=False)

    # ---- Per-cohort summary ----
    cohort_rows = []
    for cohort, group in df.groupby("cohort"):
        row = {"cohort": cohort}
        for col, label in [("dice", "dice"), ("iou", "iou"),
                            ("precision", "precision"), ("recall", "recall"),
                            ("hd95_mm", "hd95_mm")]:
            row.update(summarize(group[col].tolist(), label))
        cohort_rows.append(row)
    pd.DataFrame(cohort_rows).to_csv(metrics_dir / "cohort_metrics.csv", index=False)

    # Consolidated JSON artifact (Section 17)
    import json
    test_results = {
        "overall": overall,
        "per_cohort": cohort_rows,
        "per_case": df.to_dict(orient="records"),
    }
    with open(args.out_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Dice     : mean={overall['dice_mean']:.4f}  median={overall['dice_median']:.4f}  "
          f"IQR=[{overall['dice_iqr_low']:.4f}, {overall['dice_iqr_high']:.4f}]")
    print(f"IoU      : mean={overall['iou_mean']:.4f}  median={overall['iou_median']:.4f}")
    print(f"Precision: mean={overall['precision_mean']:.4f}")
    print(f"Recall   : mean={overall['recall_mean']:.4f}")
    if overall["hd95_mm_n"] > 0:
        print(f"HD95     : mean={overall['hd95_mm_mean']:.2f}mm  "
              f"median={overall['hd95_mm_median']:.2f}mm  (n={overall['hd95_mm_n']}/{len(df)})")
    else:
        print("HD95     : undefined for all cases")

    # ---- Plots ----
    plot_loss_curves(args.history_json, plots_dir)
    plot_dice_distribution(df, plots_dir)

    # ---- Best / median / worst qualitative cases ----
    df_sorted = df.sort_values("dice").reset_index(drop=True)
    best_row = df_sorted.iloc[-1]
    worst_row = df_sorted.iloc[0]
    median_row = df_sorted.iloc[len(df_sorted) // 2]

    for label, row in [("best", best_row), ("median", median_row), ("worst", worst_row)]:
        cid = row["case_id"]
        mri_np, gt_np, pred_np = volumes[cid]
        save_case_visualization(
            cid, row["dice"], mri_np, gt_np, pred_np,
            qual_dir / f"{label}_case_{cid}.png"
        )
        print(f"{label.capitalize():>7} case: {cid} (Dice={row['dice']:.4f}) "
              f"-> {qual_dir / (label + '_case_' + cid + '.png')}")

    print(f"\nAll results saved under: {args.out_dir}")


if __name__ == "__main__":
    main()