"""Build consolidated TerraScope experiment comparisons.

Run from the repository root:
python -m src.analysis.compare_experiments \
    --results-root results/so2sat \
    --output-root results/so2sat
"""

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ["M1", "M2", "M3", "M4", "M5"]


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def first_value(data, candidates):
    for key in candidates:
        if key in data and data[key] is not None:
            return data[key]
    return None


def collect_results(results_root):
    rows = []

    for model in MODELS:
        model_dir = Path(results_root) / model
        validation = read_json(model_dir / "validation_metrics.json")
        test = read_json(model_dir / "test_metrics.json")

        rows.append({
            "model": model,
            "validation_loss": first_value(
                validation, ["val_loss", "validation_loss"]
            ),
            "validation_accuracy": first_value(
                validation, ["val_accuracy", "accuracy"]
            ),
            "validation_macro_f1": first_value(
                validation, ["val_macro_f1", "macro_f1"]
            ),
            "validation_weighted_f1": first_value(
                validation, ["val_weighted_f1", "weighted_f1"]
            ),
            "test_loss": first_value(
                test, ["test_loss", "loss"]
            ),
            "test_accuracy": first_value(
                test, ["test_accuracy", "accuracy"]
            ),
            "test_macro_f1": first_value(
                test, ["test_macro_f1", "macro_f1"]
            ),
            "test_weighted_f1": first_value(
                test, ["test_weighted_f1", "weighted_f1"]
            ),
            "test_samples": first_value(
                test, ["test_samples", "samples", "n_samples"]
            ),
        })

    return pd.DataFrame(rows)


def build_contrasts(comparison):
    indexed = comparison.set_index("model")
    pairs = [
        ("M3", "M4", "EuroSAT path: fine-tuning effect"),
        ("M2", "M5", "ImageNet path: fine-tuning effect"),
        ("M4", "M5", "Full fine-tuning: EuroSAT minus ImageNet"),
        ("M2", "M3", "Frozen backbone: ImageNet minus EuroSAT"),
        ("M1", "M4", "Random initialization vs EuroSAT-pretrained"),
        ("M1", "M5", "Random initialization vs ImageNet-pretrained"),
    ]

    rows = []
    metrics = [
        "validation_accuracy",
        "test_accuracy",
        "validation_macro_f1",
        "test_macro_f1",
        "test_weighted_f1",
    ]

    for first, second, description in pairs:
        row = {
            "comparison": f"{second} minus {first}",
            "interpretation": description,
            "first_model": first,
            "second_model": second,
        }
        for metric in metrics:
            a = indexed.loc[first, metric]
            b = indexed.loc[second, metric]
            row[f"{metric}_difference"] = (
                float(b - a) if pd.notna(a) and pd.notna(b) else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate TerraScope validation and test metrics."
    )
    parser.add_argument("--results-root", default="results/so2sat")
    parser.add_argument("--output-root", default="results/so2sat")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    comparison = collect_results(results_root)

    comparison.to_csv(
        output_root / "experiment_comparison.csv", index=False
    )

    # Keep the established filename used by the project README.
    comparison[[
        "model", "test_loss", "test_accuracy",
        "test_macro_f1", "test_weighted_f1", "test_samples",
    ]].to_csv(
        output_root / "official_test_comparison.csv", index=False
    )

    contrasts = build_contrasts(comparison)
    contrasts.to_csv(
        output_root / "experiment_contrasts.csv", index=False
    )

    print("\nCombined validation and test results:")
    print(comparison.to_string(index=False))

    print("\nExperimental contrasts:")
    print(contrasts.to_string(index=False))

    print("\nSaved:")
    print(output_root / "experiment_comparison.csv")
    print(output_root / "official_test_comparison.csv")
    print(output_root / "experiment_contrasts.csv")


if __name__ == "__main__":
    main()
