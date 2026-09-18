"""
Backfill test_results.json (Section 17 artifact) from an already-completed
evaluate.py run, using the CSVs it already wrote -- no re-inference needed.

Usage (run once per results directory):
    python export_test_results_json.py --results-dir "C:\\...\\results"
    python export_test_results_json.py --results-dir "C:\\...\\results_no_qnn"
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="Directory containing metrics/case_metrics.csv, "
                              "metrics/test_metrics_summary.csv, metrics/cohort_metrics.csv "
                              "(i.e. the --out-dir you passed to evaluate.py)")
    args = parser.parse_args()

    metrics_dir = args.results_dir / "metrics"
    case_df = pd.read_csv(metrics_dir / "case_metrics.csv")
    summary_df = pd.read_csv(metrics_dir / "test_metrics_summary.csv")
    cohort_df = pd.read_csv(metrics_dir / "cohort_metrics.csv")

    result = {
        "overall": summary_df.iloc[0].to_dict(),
        "per_cohort": cohort_df.to_dict(orient="records"),
        "per_case": case_df.to_dict(orient="records"),
    }

    out_path = args.results_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()