"""
Extract misclassified examples from manual evaluation predictions.
Saves manual_errors_only.csv (all errors) and high_missed_cases.csv (true high predicted as low/med).
Prioritizes true high-risk cases that were missed.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
PREDICTIONS_CSV = REPORTS_DIR / "manual_eval_predictions.csv"
ERRORS_CSV = REPORTS_DIR / "manual_errors_only.csv"
HIGH_MISSED_CSV = REPORTS_DIR / "high_missed_cases.csv"


def main():
    if len(sys.argv) > 1:
        predictions_path = Path(sys.argv[1])
    else:
        predictions_path = PREDICTIONS_CSV

    if not predictions_path.exists():
        print(f"File not found: {predictions_path}")
        print("Run: python scripts/eval_manual_set.py first.")
        sys.exit(1)

    df = pd.read_csv(predictions_path)
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        print("Expected columns: text, true_label, pred_label, pred_score, correct")
        sys.exit(1)

    # All misclassified
    errors = df[df["correct"] == False].copy()
    errors = errors.reset_index(drop=True)

    # True high that were predicted low or med (prioritized)
    high_missed = df[(df["true_label"] == "high") & (df["pred_label"] != "high")].copy()
    high_missed = high_missed.reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERRORS_CSV, index=False)
    high_missed.to_csv(HIGH_MISSED_CSV, index=False)

    print("Manual evaluation errors")
    print("=" * 50)
    print(f"Predictions file: {predictions_path}")
    print(f"Total misclassified: {len(errors)}")
    print(f"True high missed (predicted low/med): {len(high_missed)}")
    print(f"Saved: {ERRORS_CSV}")
    print(f"Saved: {HIGH_MISSED_CSV}")
    if len(high_missed) > 0:
        print("\nHigh-risk cases missed (first 5):")
        for i, row in high_missed.head(5).iterrows():
            print(f"  true={row['true_label']} pred={row['pred_label']} score={row.get('pred_score', '')}")
            print(f"    {str(row.get('text', ''))[:80]}...")


if __name__ == "__main__":
    main()
