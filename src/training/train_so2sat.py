
import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data.so2sat import (
    So2SatDataset,
    load_so2sat_split_indices,
)
from src.models.resnet import (
    build_resnet18,
    load_eurosat_backbone,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "LCZ1 Compact high-rise",
    "LCZ2 Compact midrise",
    "LCZ3 Compact low-rise",
    "LCZ4 Open high-rise",
    "LCZ5 Open midrise",
    "LCZ6 Open low-rise",
    "LCZ7 Lightweight low-rise",
    "LCZ8 Large low-rise",
    "LCZ9 Sparsely built",
    "LCZ10 Heavy industry",
    "LCZA Dense trees",
    "LCZB Scattered trees",
    "LCZC Bush/scrub",
    "LCZD Low plants",
    "LCZE Bare rock or paved",
    "LCZF Bare soil or sand",
    "LCZG Water",
]

NUM_CLASSES = 17

DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_experiment(experiment, eurosat_checkpoint, device):

    print(f"\nInitializing {experiment}...")

    if experiment == "M1":
        # Random initialization, train the entire network.
        model = build_resnet18(
            num_classes=NUM_CLASSES,
            initialization="scratch",
        )
        frozen = False

    elif experiment == "M2":
        # ImageNet initialization, freeze the backbone.
        model = build_resnet18(
            num_classes=NUM_CLASSES,
            initialization="imagenet",
        )
        frozen = True

    elif experiment == "M3":
        # EuroSAT-pretrained backbone, freeze the backbone.
        model = build_resnet18(
            num_classes=NUM_CLASSES,
            initialization="scratch",
        )
        load_eurosat_backbone(model, eurosat_checkpoint)
        frozen = True

    elif experiment == "M4":
        # EuroSAT-pretrained backbone, fine-tune the entire network.
        model = build_resnet18(
            num_classes=NUM_CLASSES,
            initialization="scratch",
        )
        load_eurosat_backbone(model, eurosat_checkpoint)
        frozen = False

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    model = model.to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen backbone: {frozen}")

    return model, frozen


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model, frozen):

    if frozen:
        optimizer = torch.optim.AdamW(
            model.fc.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
        )

    else:
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("fc.")
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": BACKBONE_LR,
                },
                {
                    "params": model.fc.parameters(),
                    "lr": HEAD_LR,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )

    return optimizer


# ============================================================
# TRAINING MODE
# ============================================================

def set_training_mode(model, frozen):

    model.train()

    if frozen:
        # Keep backbone BatchNorm running statistics unchanged.
        model.eval()

        # The classifier remains trainable.
        model.fc.train()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "val_loss": float(average_loss),
        "val_accuracy": float(accuracy),
        "val_macro_f1": float(macro_f1),
        "val_weighted_f1": float(weighted_f1),
    }

    return metrics, all_labels, all_predictions


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    frozen,
):

    set_training_mode(model, frozen)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return float(average_loss), float(accuracy)


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

def save_validation_report(
    model,
    loader,
    criterion,
    device,
    output_dir,
):

    metrics, labels, predictions = evaluate(
        model,
        loader,
        criterion,
        device,
    )

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    with open(
        os.path.join(output_dir, "validation_metrics.json"),
        "w",
    ) as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(output_dir, "classification_report.csv")
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        os.path.join(output_dir, "confusion_matrix.csv")
    )

    print("\nFinal validation results")
    print("------------------------")

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def train_experiment(args):

    experiment = args.experiment

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"TERRASCOPE AI: {experiment}")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # LOAD SPLIT INDICES
    # --------------------------------------------------------

    train_indices, val_indices = load_so2sat_split_indices(
        args.split_path
    )

    print(f"\nTraining samples: {len(train_indices):,}")
    print(f"Validation samples: {len(val_indices):,}")

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    train_dataset = So2SatDataset(
        h5_path=args.validation_h5,
        indices=train_indices,
    )

    val_dataset = So2SatDataset(
        h5_path=args.validation_h5,
        indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------------------------

    model, frozen = initialize_experiment(
        experiment,
        args.eurosat_checkpoint,
        device,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model,
        frozen,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            frozen,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_metrics["val_loss"])

        learning_rates = [
            group["lr"] for group in optimizer.param_groups
        ]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            **val_metrics,
            "learning_rates": learning_rates,
        }

        history.append(epoch_record)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        print(f"Train loss:      {train_loss:.6f}")
        print(f"Train accuracy:  {train_accuracy:.6f}")
        print(f"Val loss:        {val_metrics['val_loss']:.6f}")
        print(f"Val accuracy:    {val_metrics['val_accuracy']:.6f}")
        print(f"Val macro-F1:    {val_metrics['val_macro_f1']:.6f}")
        print(f"Val weighted-F1: {val_metrics['val_weighted_f1']:.6f}")

        print(f"Learning rates:  {learning_rates}")
        print(f"Time:            {elapsed:.1f} seconds")

        # Save only the best checkpoint to limit Drive storage.
        if val_metrics["val_loss"] < best_val_loss:

            best_val_loss = val_metrics["val_loss"]

            checkpoint = {
                "experiment": experiment,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": epoch_record,
                "best_val_loss": best_val_loss,
                "architecture": "ResNet18",
                "num_classes": NUM_CLASSES,
                "seed": args.seed,
            }

            checkpoint_path = os.path.join(
                args.output_dir,
                "best_checkpoint.pth",
            )

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print("New best checkpoint saved.")

        # Save history after every epoch in case the runtime disconnects.
        with open(
            os.path.join(args.output_dir, "training_history.json"),
            "w",
        ) as file:
            json.dump(history, file, indent=4)

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        args.output_dir,
        "best_checkpoint.pth",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nBest checkpoint restored.")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # --------------------------------------------------------
    # FINAL VALIDATION REPORT
    # --------------------------------------------------------

    save_validation_report(
        model,
        val_loader,
        criterion,
        device,
        args.output_dir,
    )

    print("\n" + "=" * 60)
    print(f"{experiment} COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

    # Close HDF5 files when possible.
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train TerraScope So2Sat experiments."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["M1", "M2", "M3", "M4"],
    )

    parser.add_argument(
        "--validation-h5",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--eurosat-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    train_experiment(args)


if __name__ == "__main__":
    main()
