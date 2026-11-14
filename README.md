# TerraScope AI

**Investigating Geographic Transfer Learning in Satellite Image Classification**

## Overview

TerraScope AI investigates whether pretraining on EuroSAT improves satellite-image classification on the So2Sat LCZ42 dataset, and whether fine-tuning pretrained representations provides additional benefits.

The project evaluates five ResNet-18 configurations to distinguish the effects of initialization, pretraining history, and target-dataset fine-tuning.

## Research Question

Does pretraining on EuroSAT improve performance on a geographically different dataset, and how much additional benefit does fine-tuning provide?

## Datasets

### EuroSAT
A Sentinel-2 satellite-image dataset containing 10 land-use and land-cover classes. It is used to develop pretrained representations.

### So2Sat LCZ42
A satellite-image dataset containing Sentinel-1 and Sentinel-2 observations labeled across 17 Local Climate Zone (LCZ) classes. It is used as the target classification task.

The experiments use 32 × 32 RGB inputs derived from Sentinel-2 bands B4, B3, and B2.

## Experimental Design

All five experiments use ResNet-18 and the same So2Sat training and validation split.

| Experiment | Initialization | Target-dataset training |
|---|---|---|
| M1 | Random initialization | Full fine-tuning |
| M2 | ImageNet pretrained | Frozen backbone |
| M3 | ImageNet → EuroSAT pretrained | Frozen backbone |
| M4 | ImageNet → EuroSAT pretrained | Full fine-tuning |
| M5 | ImageNet pretrained | Full fine-tuning |

The comparisons address different questions:

- **M2 vs. M3:** Does EuroSAT pretraining change the usefulness of frozen representations relative to ImageNet initialization?
- **M3 vs. M4:** What is the effect of fine-tuning EuroSAT-pretrained representations?
- **M4 vs. M5:** How do EuroSAT and ImageNet initialization compare when both models are fully fine-tuned under the same target-dataset protocol?
- **M2 vs. M5:** What is the effect of fine-tuning an ImageNet-pretrained model?

Comparisons that change more than one experimental factor are interpreted cautiously.

## Data Partitioning

The official So2Sat validation partition contains 24,119 samples. It was internally divided into:

- Training subset: 19,295 samples
- Validation subset: 4,824 samples

The split was fixed and shared across all five experiments.

The official testing partition, containing 24,188 samples, is reserved for final evaluation. It is not used for model training, checkpoint selection, or hyperparameter adjustment.

The validation and testing partitions cover the same ten cities. Therefore, this study examines cross-dataset transfer and generalization to held-out samples, not generalization to entirely unseen cities.

## Training Protocol

The So2Sat experiments use a consistent training protocol:

- Architecture: ResNet-18
- Epochs: 12
- Batch size: 64
- Random seed: 42
- Optimizer: AdamW
- Weight decay: 0.0001
- Backbone learning rate: 0.0001
- Classification-head learning rate: 0.001
- Checkpoint selection: lowest validation loss

The frozen-backbone experiments train the classification head while keeping the pretrained feature extractor frozen.

## Evaluation

The project evaluates:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class precision, recall, and F1
- Confusion matrices
- Training and validation histories

Macro-F1 is particularly important because the 17 LCZ classes are imbalanced.

## Preliminary Validation Results

The best validation results from each experiment are:

| Experiment | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| M1 | 68.99% | 0.5950 | 0.6899 |
| M2 | 59.02% | 0.4346 | 0.5701 |
| M3 | 51.14% | 0.3292 | 0.4672 |
| M4 | 92.04% | 0.8935 | 0.9198 |
| M5 | 92.23% | 0.8945 | 0.9216 |

These are validation results, not final test results.

M4 and M5 achieved substantially stronger validation performance than the other configurations. Their results were also very close: M5 had slightly higher accuracy and F1 scores, while M4 had slightly lower validation loss.

The official testing partition will be used to assess whether these findings hold on unseen samples.

Detailed experiment outputs are available in `results/so2sat/`.

## Technology

- Python
- PyTorch
- torchvision
- NumPy
- h5py
- scikit-learn
- Matplotlib

## Repository Structure

- `configs/`: Experiment and preprocessing configurations.
- `src/data/`: Dataset loading and preprocessing.
- `src/models/`: Model architecture.
- `src/training/`: EuroSAT and So2Sat training scripts.
- `results/`: Experiment histories and validation outputs.
- `notebooks/`: Exploratory and development notebooks.

## Reproducibility and Storage

Source code, configurations, and compact experiment results are maintained in this repository.

Original datasets and model checkpoints are stored separately because of their size. The data files and checkpoint files are not committed to GitHub.

## Project Status

- [x] Dataset exploration and preprocessing
- [x] EuroSAT pretraining
- [x] So2Sat experiments M1–M5
- [x] Validation analysis
- [ ] Evaluation on the official So2Sat testing partition
- [ ] Final cross-experiment comparison
- [ ] Final documentation and presentation

## Limitations

The study uses a single fixed train-validation split and one primary random seed. Results may therefore be sensitive to the particular partition and initialization.

The official validation and testing partitions cover the same ten cities, so the study does not establish generalization to entirely unseen geographic regions or cities.

The class distribution is imbalanced, making accuracy alone insufficient to characterize model performance. Macro-F1 and per-class metrics are reported alongside accuracy.
