"""
Training script (Section 13). Reads the preprocessed cache + split_seed.json,
trains HybridSegModel(use_qnn=True) with DiceCELoss, tracks validation Dice,
saves the best checkpoint, and writes training_history.json.

Run the Stage 1/2/3 pre-flight checks BEFORE this. This script assumes
they already passed.

Usage:
    python train.py --cache-dir "...\\MAMA_MIA_100_preprocessed_128" ^
                     --split-json "...\\split_seed.json" ^
                     --out-dir "...\\training_output"
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from model import HybridSegModel
from dataset import MamaMiaDataset
from metrics import dice_score


def set_determinism(seed=42):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn_only: some 3D conv/upsample
    torch.backends.cudnn.deterministic = True                  # ops lack deterministic CUDA kernels
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_dice, n_batches, n_cases = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                dice = dice_score(probs, y)  # (B,)

            total_loss += loss.item() * x.shape[0]
            total_dice += dice.sum().item()
            n_batches += 1
            n_cases += x.shape[0]

    return total_loss / n_cases, total_dice / n_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15,
                         help="Early stopping patience, in epochs with no val Dice improvement")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--use-qnn", action="store_true", default=True)
    parser.add_argument("--no-qnn", dest="use_qnn", action="store_false",
                         help="Ablation switch: disable the QNN bottleneck")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_determinism(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU. This will be very slow for 3D volumes -- "
              "confirm CUDA is set up correctly before a full run.")

    train_ds = MamaMiaDataset(args.split_json, args.cache_dir, split="train")
    val_ds = MamaMiaDataset(args.split_json, args.cache_dir, split="val")
    print(f"Train cases: {len(train_ds)}  |  Val cases: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = HybridSegModel(
        in_channels=1, base_channels=args.base_channels, out_channels=1,
        use_qnn=args.use_qnn,
    ).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": [], "lr": []}
    best_val_dice = -1.0
    epochs_no_improve = 0
    best_ckpt_path = args.out_dir / "best_model.pt"

    print(f"\nStarting training: {args.epochs} max epochs, patience={args.patience}, "
          f"use_qnn={args.use_qnn}")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
              f"lr={current_lr:.2e} | {time.time()-t_epoch:.1f}s"
              f"{'  <-- best' if improved else ''}")

        with open(args.out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no val Dice improvement for {args.patience} epochs.")
            break

    total_time = time.time() - t_start
    print(f"\nTraining finished in {total_time/60:.1f} min. "
          f"Best val Dice: {best_val_dice:.4f} (checkpoint: {best_ckpt_path})")


if __name__ == "__main__":
    main()