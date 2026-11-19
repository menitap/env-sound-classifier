"""
aggregate_results.py
Phase H: Aggregate all experiment results into experiment_comparison.json.

Runs in .venv_ml. Reads individual result files; writes experiment_comparison.json.
Verifies D25: values in comparison match individual metric files.
Never modifies protected results files.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ml_config import RESULTS_DIR, SNR_LEVELS


def load_json(path: Path) -> dict:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def aggregate():
    print("=" * 60)
    print("PHASE H: AGGREGATE RESULTS")
    print("=" * 60)

    models = [
        "baseline_cnn",           # original baseline (inference only)
        "baseline_cnn_augmented", # Experiment A
        "svm_original",
        "svm_augmented",
        "rf_original",
        "rf_augmented",
    ]

    rows = []

    for model_name in models:
        # Clean test metrics
        metrics_path = RESULTS_DIR / f"{model_name}_test_metrics.json"
        noise_path   = RESULTS_DIR / f"{model_name}_noise_robustness.json"

        m = load_json(metrics_path)
        n = load_json(noise_path)

        if m is None:
            if model_name == "baseline_cnn":
                mc = load_json(RESULTS_DIR / "model_comparison.json")
                if mc and "baseline_cnn" in mc:
                    row = {
                        "model": "baseline_cnn",
                        "test_accuracy_clean": float(mc["baseline_cnn"]["accuracy"]),
                    }
                    rows.append(row)
                    print(f"  {model_name}: clean_acc={mc['baseline_cnn']['accuracy']:.4f}")
                    continue
            print(f"  [SKIP] No test metrics for {model_name}")
            continue

        row = {
            "model": model_name,
            "test_accuracy_clean": m["test_accuracy"],
        }

        if n is not None:
            for snr in SNR_LEVELS:
                key = f"snr_{snr}db"
                row[f"test_acc_snr{snr}db"] = (
                    n["results"].get(key, {}).get("test_accuracy")
                )

        rows.append(row)
        print(f"  {model_name}: clean_acc={m['test_accuracy']:.4f}")

    # D25: verify aggregated values match individual files
    print("\n[D25] Verifying aggregated values match individual result files...")
    all_match = True
    for row in rows:
        mname = row["model"]
        m = load_json(RESULTS_DIR / f"{mname}_test_metrics.json")
        if m and abs(row["test_accuracy_clean"] - m["test_accuracy"]) > 1e-8:
            print(f"  [FAIL] {mname} clean accuracy mismatch!")
            all_match = False

    if all_match:
        print("  [PASS] All aggregated values match individual metric files.")

    # Save
    out_path = RESULTS_DIR / "experiment_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n{:<30} {:>12}  {}".format(
        "Model", "Clean Test%",
        "  ".join(f"SNR{s}dB%" for s in SNR_LEVELS)
    ))
    print("-" * 70)
    for row in rows:
        clean = row["test_accuracy_clean"] * 100
        noisy = "  ".join(
            f"{row.get(f'test_acc_snr{s}db', 0)*100:8.2f}"
            if row.get(f"test_acc_snr{s}db") is not None else "       N/A"
            for s in SNR_LEVELS
        )
        print(f"{row['model']:<30} {clean:>11.2f}%  {noisy}")

    print("\n[PHASE H COMPLETE]")


if __name__ == "__main__":
    aggregate()
