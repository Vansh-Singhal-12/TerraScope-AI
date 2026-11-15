"""Evaluate saved TerraScope checkpoints on the official So2Sat test set.

Example:
python -m src.evaluation.evaluate_so2sat \
    --test-h5 data/raw/testing.h5 \
    --checkpoint-root /path/to/checkpoints/so2sat \
    --output-root results/so2sat
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from torch.utils.data import DataLoader

from src.data.so2sat import So2SatDataset
from src.models.resnet import build_resnet18


CLASS_NAMES = [
    "LCZ1 Compact high-rise", "LCZ2 Compact midrise",
    "LCZ3 Compact low-rise", "LCZ4 Open high-rise",
    "LCZ5 Open midrise", "LCZ6 Open low-rise",
    "LCZ7 Lightweight low-rise", "LCZ8 Large low-rise",
    "LCZ9 Sparsely built", "LCZ10 Heavy industry",
    "LCZA Dense trees", "LCZB Scattered trees",
    "LCZC Bush/scrub", "LCZD Low plants",
    "LCZE Bare rock or paved", "LCZF Bare soil or sand",
    "LCZG Water",
]
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Missing model_state_dict: {checkpoint_path}")

    model = build_resnet18(NUM_CLASSES, initialization="scratch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def predict(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    labels_all, predictions_all = [], []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(logits.argmax(1).cpu().numpy())

    labels_all = np.asarray(labels_all, dtype=np.int64)
    predictions_all = np.asarray(predictions_all, dtype=np.int64)

    metrics = {
        "test_loss": float(total_loss / len(loader.dataset)),
        "test_accuracy": float(accuracy_score(labels_all, predictions_all)),
        "test_macro_f1": float(f1_score(
            labels_all, predictions_all,
            labels=np.arange(NUM_CLASSES),
            average="macro", zero_division=0,
        )),
        "test_weighted_f1": float(f1_score(
            labels_all, predictions_all,
            labels=np.arange(NUM_CLASSES),
            average="weighted", zero_division=0,
        )),
        "test_samples": int(len(labels_all)),
    }
    return metrics, labels_all, predictions_all


def evaluate_experiment(
    experiment, test_h5, checkpoint_root, output_root, batch_size, device
):
    checkpoint_path = (
        Path(checkpoint_root) / experiment / "best_checkpoint.pth"
    )
    output_dir = Path(output_root) / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {experiment}")
    model, checkpoint = load_model(checkpoint_path, device)

    with h5py.File(test_h5, "r") as handle:
        sample_count = int(handle["label"].shape[0])

    dataset = So2SatDataset(
        h5_path=str(test_h5),
        indices=np.arange(sample_count, dtype=np.int64),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    metrics, labels, predictions = predict(model, loader, device)
    metrics.update({
        "experiment": experiment,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_path": str(checkpoint_path),
    })

    report = classification_report(
        labels, predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(NUM_CLASSES)
    )

    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "test_classification_report.csv"
    )
    pd.DataFrame(
        matrix, index=CLASS_NAMES, columns=CLASS_NAMES
    ).to_csv(output_dir / "test_confusion_matrix.csv")

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        labels=labels,
        predictions=predictions,
        y_true=labels,
        y_pred=predictions,
    )

    print(json.dumps(metrics, indent=2))
    del model, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TerraScope checkpoints on the official test set."
    )
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", default="results/so2sat")
    parser.add_argument(
        "--experiments", nargs="+",
        default=["M1", "M2", "M3", "M4", "M5"],
        choices=["M1", "M2", "M3", "M4", "M5"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        device = torch.device(args.device)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print("Device:", device)

    results = [
        evaluate_experiment(
            experiment, args.test_h5, args.checkpoint_root,
            output_root, args.batch_size, device,
        )
        for experiment in args.experiments
    ]

    comparison = pd.DataFrame(results)
    comparison.to_csv(output_root / "official_test_comparison.csv", index=False)
    print("\nOfficial test comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
