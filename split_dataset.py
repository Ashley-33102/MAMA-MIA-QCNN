"""
Train/val/test split for the MAMA-MIA 100-case dataset (Section 6).

Hardcoded per-cohort split (default 17 train / 4 val / 4 test), fixed seed,
reads usable case IDs from the preprocessing report so any FAIL cases are
automatically excluded and WARN (small-tumor) cases are included by default.

Usage:
    python split_dataset.py --report "C:\\...\\preprocessing_output\\preprocessing_report_128.csv"
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True,
                         help="Path to preprocessing_report_<N>.csv")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output JSON path (default: split_seed.json next to the report)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=17)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--exclude-warn", action="store_true",
                         help="Exclude small-tumor WARN cases from the split entirely "
                              "(default: keep them in, they're valid cases, just flagged)")
    args = parser.parse_args()

    df = pd.read_csv(args.report)
    keep_statuses = ("OK",) if args.exclude_warn else ("OK", "WARN")
    df = df[df["status"].astype(str).str.startswith(keep_statuses)].copy()

    n_dropped = 0
    if "status" in df.columns:
        all_df = pd.read_csv(args.report)
        n_dropped = len(all_df) - len(df)
    if n_dropped:
        print(f"Excluded {n_dropped} case(s) not matching {keep_statuses} (FAIL/ERROR"
              f"{'and WARN' if args.exclude_warn else ''}).")

    out_path = args.out or (args.report.parent / "split_seed.json")

    rng = random.Random(args.seed)
    split = {"train": [], "val": [], "test": []}
    per_cohort_actual = {}

    for cohort, group in df.groupby("cohort"):
        case_ids = sorted(group["case_id"].tolist())
        rng.shuffle(case_ids)
        n = len(case_ids)
        n_train, n_val, n_test = args.train, args.val, args.test

        if n < n_train + n_val + n_test:
            print(f"WARNING: cohort {cohort} has only {n} usable cases, fewer than the "
                  f"requested {n_train + n_val + n_test}. Falling back to a proportional "
                  f"split for this cohort and documenting the actual counts below.")
            n_test = min(n_test, max(1, n // 5))
            n_val = min(n_val, max(1, n // 5))
            n_train = max(0, n - n_val - n_test)

        train_ids = case_ids[:n_train]
        val_ids = case_ids[n_train:n_train + n_val]
        test_ids = case_ids[n_train + n_val: n_train + n_val + n_test]

        split["train"].extend(train_ids)
        split["val"].extend(val_ids)
        split["test"].extend(test_ids)
        per_cohort_actual[cohort] = {
            "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
            "total_usable": n,
        }

    result = {
        "seed": args.seed,
        "requested_per_cohort": {"train": args.train, "val": args.val, "test": args.test},
        "actual_per_cohort": per_cohort_actual,
        "split": split,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    for cohort, counts in per_cohort_actual.items():
        print(f"{cohort:>8}: train={counts['train']:2d}  val={counts['val']:2d}  "
              f"test={counts['test']:2d}  (usable={counts['total_usable']})")
    print(f"\nTotal: train={len(split['train'])}  val={len(split['val'])}  "
          f"test={len(split['test'])}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()