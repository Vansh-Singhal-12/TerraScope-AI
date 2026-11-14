# Results

This directory contains training histories and evaluation outputs for TerraScope AI.

## EuroSAT Pretraining

- `eurosat_training_history.json`: Training and validation history from EuroSAT pretraining.

## So2Sat Experiments

Each experiment directory under `so2sat/` contains:

| File | Description |
|---|---|
| `validation_metrics.json` | Aggregate validation metrics |
| `classification_report.csv` | Per-class precision, recall, F1, and support |
| `confusion_matrix.csv` | Validation confusion matrix |
| `training_history.json` | Epoch-level training and validation history |

The five experiments are:

- **M1:** Random initialization, full fine-tuning.
- **M2:** ImageNet initialization, frozen backbone.
- **M3:** ImageNet-to-EuroSAT pretraining, frozen backbone.
- **M4:** ImageNet-to-EuroSAT pretraining, full fine-tuning.
- **M5:** ImageNet initialization, full fine-tuning.

## Validation Summary

| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| M1 | 0.6899 | 0.5950 | 0.6899 |
| M2 | 0.5902 | 0.4346 | 0.5701 |
| M3 | 0.5114 | 0.3292 | 0.4672 |
| M4 | 0.9204 | 0.8935 | 0.9198 |
| M5 | 0.9223 | 0.8945 | 0.9216 |

These metrics come from the internal validation subset, not the official testing partition.

## Model Checkpoints

Large model checkpoints (`.pth`) are intentionally excluded from this repository. They are stored separately alongside the datasets.

## Final Testing

The official So2Sat testing partition is reserved for final evaluation. Testing metrics and associated analysis will be added here after that evaluation is completed.
