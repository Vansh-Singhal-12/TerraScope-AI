"""EuroSAT pretraining entry point.

This script records the training procedure used for the EuroSAT source
model. It supports CUDA mixed precision and saves an epoch checkpoint,
best checkpoint (selected by validation loss), and JSON history.

Example:
    python -m src.training.train_eurosat \
        --data-root ./data \
        --checkpoint-dir ./checkpoints \
        --epochs 12
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.eurosat import create_eurosat_loaders
from src.models.resnet import build_resnet18


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []

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

            total_loss += loss.item() * images.size(0)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "weighted_f1": f1_score(targets, predictions, average="weighted"),
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, _, _ = create_eurosat_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        random_state=args.seed,
        num_workers=args.num_workers,
    )

    model = build_resnet18(
        num_classes=10,
        initialization=args.initialization,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer1.parameters(), "lr": 1e-4},
            {"params": model.layer2.parameters(), "lr": 1e-4},
            {"params": model.layer3.parameters(), "lr": 1e-4},
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.conv1.parameters(), "lr": 1e-4},
            {"params": model.bn1.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        train_predictions, train_targets = [], []

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"
        ):
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

            total_train_loss += loss.item() * images.size(0)
            train_predictions.extend(outputs.argmax(1).detach().cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_metrics = {
            "loss": total_train_loss / len(train_loader.dataset),
            "accuracy": accuracy_score(train_targets, train_predictions),
        }
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": record,
            "architecture": "ResNet18 adapted for 32x32 images",
            "num_classes": 10,
        }

        epoch_path = os.path.join(
            args.checkpoint_dir, f"eurosat_epoch_{epoch + 1}.pth"
        )
        temp_path = epoch_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, epoch_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(
                args.checkpoint_dir, "eurosat_resnet18_best.pth"
            )
            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "metrics": record,
                "best_val_loss": best_val_loss,
                "architecture": "ResNet18 adapted for 32x32 images",
                "num_classes": 10,
            }
            temp_best = best_path + ".tmp"
            torch.save(best_checkpoint, temp_best)
            os.replace(temp_best, best_path)

        history_path = os.path.join(
            args.checkpoint_dir, "eurosat_training_history.json"
        )
        temp_history = history_path + ".tmp"
        with open(temp_history, "w") as handle:
            json.dump(history, handle, indent=2)
        os.replace(temp_history, history_path)

        print(record)

    print("Training complete. Best validation loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--initialization",
        choices=["imagenet", "scratch"],
        default="imagenet",
    )
    main(parser.parse_args())
