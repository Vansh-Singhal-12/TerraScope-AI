# TerraScope AI
### Satellite Image Classification Through Transfer Learning

## Overview

TerraScope AI is a deep-learning project that classifies satellite imagery into 17 Local Climate Zone (LCZ) categories using Sentinel-2 data.

Beyond developing an accurate image classifier, this project investigates how transfer learning affects satellite-image classification. In particular, we examine whether pretrained visual representations improve classification, whether additional satellite-specific pretraining provides benefits, and how fine-tuning influences transfer performance.

We compare models initialized randomly, pretrained on ImageNet, and pretrained sequentially on ImageNet and EuroSAT before adapting them to So2Sat.

## Research Question

**To what extent does transfer learning improve satellite-image classification, and how do pretraining pathways influence classification performance and spatial generalization?**

### Research Objectives

1. Evaluate the effect of transfer learning compared with training a model from random initialization.
2. Investigate whether additional satellite-specific pretraining on EuroSAT improves transfer performance over ImageNet initialization alone.
3. Examine how freezing or fine-tuning pretrained feature extractors affects classification performance.
4. Evaluate whether these differences persist on held-out So2Sat data and investigate geographic separation within the represented cities.

## Datasets

### ImageNet

ImageNet provides general-purpose visual representations through large-scale natural-image pretraining. We use an ImageNet-pretrained ResNet-18 as one initialization pathway.

### EuroSAT

EuroSAT is a satellite-image land-cover classification dataset containing 10 classes. We use it for additional domain-specific pretraining after ImageNet initialization.

### So2Sat LCZ42

So2Sat is the target dataset. It contains Sentinel-2 satellite imagery associated with 17 Local Climate Zone categories.

- Input: Sentinel-2 imagery
- Selected bands: B4, B3, B2 (RGB)
- Input dimensions: 32 × 32 pixels
- Number of target classes: 17

The official So2Sat validation partition contains 24,119 samples. We divided it into an 80/20 training and validation split:

- Training: 19,295 samples
- Validation: 4,824 samples

The official testing partition contains 24,188 samples and is reserved for final evaluation.

## Experimental Design

We implemented five ResNet-18 experimental configurations.

| Model | Initialization pathway | So2Sat training |
|---|---|---|
| M1 | Random initialization | Full fine-tuning |
| M2 | ImageNet | Frozen backbone |
| M3 | ImageNet → EuroSAT | Frozen backbone |
| M4 | ImageNet → EuroSAT | Full fine-tuning |
| M5 | ImageNet | Full fine-tuning |

The original ImageNet classifier is replaced with a new 17-class LCZ classification head.

### Experimental Comparisons

| Comparison | Purpose |
|---|---|
| M1 vs. M4 / M5 | Compare random initialization with pretrained initialization |
| M2 vs. M3 | Examine the effect of additional EuroSAT pretraining with a frozen backbone |
| M3 vs. M4 | Examine the effect of fine-tuning after EuroSAT pretraining |
| M2 vs. M5 | Examine the effect of fine-tuning ImageNet-pretrained features |
| M4 vs. M5 | Compare the two pretraining pathways under full fine-tuning |

M4 versus M5 is the key comparison for investigating whether additional satellite-specific pretraining provides an advantage when both models are fully adapted to So2Sat.

## Training Protocol

All So2Sat experiments use the same target training and validation split and a standardized training configuration.

- Architecture: ResNet-18
- Input resolution: 32 × 32
- Batch size: 64
- Epochs: 12
- Random seed: 42
- Optimizer: AdamW
- Weight decay: 0.0001
- Backbone learning rate: 0.0001
- Classification head learning rate: 0.001
- Learning-rate scheduler: ReduceLROnPlateau
- Checkpoint selection: Lowest validation loss
- Mixed precision: Enabled on CUDA

For frozen-backbone experiments, the feature extractor remains frozen while the classification head is trained.

## Evaluation

We evaluate models using:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class precision, recall, and F1-score
- Confusion matrices
- Training and validation histories

Macro-F1 is particularly important because the 17 LCZ categories are not equally represented.

## Preliminary Validation Results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| M1 | 68.99% | 59.50% | 68.99% |
| M2 | 59.02% | 43.46% | 57.01% |
| M3 | 51.14% | 32.92% | 46.72% |
| M4 | 92.04% | 89.35% | 91.98% |
| M5 | 92.23% | 89.45% | 92.16% |

These are internal validation results, not final test-set results.

### Preliminary Findings

- Both fully fine-tuned pretrained models achieve over 92% validation accuracy.
- Fine-tuning substantially improves performance compared with frozen feature extraction.
- ImageNet initialization outperforms the ImageNet-to-EuroSAT pathway in the frozen-backbone comparison.
- After full fine-tuning, M4 and M5 achieve nearly identical validation performance.


## Geographic Transfer and Limitations

EuroSAT and So2Sat represent different datasets and image domains, allowing us to investigate cross-dataset transfer through satellite-specific pretraining. This is distinct from the geographic generalization audit, which examines spatial separation between So2Sat partitions.

However, the official So2Sat validation and testing partitions cover the same ten cities. Consequently, this experiment does not establish generalization to entirely unseen cities.

Other limitations include:

- A single random seed and training/validation split.
- A limited set of pretraining pathways and model configurations.
- Class imbalance across the 17 LCZ categories.
- The use of RGB bands rather than the full Sentinel-2 spectral information.

These limitations are considered when interpreting the results.


## Official Test Evaluation

All five experimental configurations were evaluated on the 24,188-sample
official So2Sat testing partition. This partition was not used for model
selection.

| Model | Initialization | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---:|---:|---:|
| M1 | Random, full fine-tuning | 60.76% | 49.22% | 60.24% |
| M2 | ImageNet, frozen backbone | 57.69% | 42.76% | 55.57% |
| M3 | ImageNet → EuroSAT, frozen backbone | 50.01% | 32.36% | 45.57% |
| M4 | ImageNet → EuroSAT, full fine-tuning | 77.14% | 69.81% | 76.58% |
| M5 | ImageNet, full fine-tuning | 76.81% | 69.95% | 76.37% |

M4 and M5 achieved very similar test performance. M4 had slightly higher
accuracy and weighted-F1, while M5 had slightly higher macro-F1. These
differences are small and do not establish a definitive advantage for
either pretraining pathway.

Fine-tuning substantially improved performance relative to the
corresponding frozen-backbone configurations. M4 outperformed M3 by
27.13 percentage points in test accuracy, while M5 outperformed M2 by
19.12 percentage points.

The EuroSAT checkpoints used for M3 and M4 were initialized
with ImageNet-pretrained weights before EuroSAT pretraining. Their complete
pretraining pathway is therefore ImageNet → EuroSAT → So2Sat.

## Geographic Generalization

The internal validation split achieved substantially higher performance
than the official test partition for the fully fine-tuned models:

- M4: 92.04% validation accuracy versus 77.14% test accuracy.
- M5: 92.23% validation accuracy versus 76.81% test accuracy.

A spatial audit found that 97.70% of internal-validation patches intersected
at least one training-patch footprint, compared with 8.28% of official-test
patches. Median distance to the nearest training-patch center was 100 m
for internal validation and 2,342.01 m for the official test partition.

These findings indicate substantially greater spatial association between
training and internal-validation samples. Spatial dependence may contribute
to the validation-to-test performance gap, although the analysis does not
establish causation.

Both official partitions contain samples from the same ten cities.
Consequently, this experiment evaluates geographic separation within
represented cities, not generalization to entirely unseen cities.

The full analysis, methodology, additional metrics, and limitations are
documented in [`docs/generalization_analysis.md`](docs/generalization_analysis.md).


## Reproducibility and Setup

### Environment

The experiments were developed in Google Colab using the following environment:

| Package | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1 |
| Torchvision | 0.29.1 |
| NumPy | 2.4.6 |
| scikit-learn | 1.6.1 |
| h5py | 3.14.0 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.0 |
| SciPy | 1.16.3 |
| tqdm | 4.67.3 |
| Pillow | 11.3.0 |

The recorded PyTorch environment used CUDA 13.0 builds. GPU availability depends on the installed PyTorch build and hardware.

### Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/Vansh-Singhal-12/TerraScope-AI.git
cd TerraScope-AI
pip install -r requirements.txt
```

For GPU training, install a compatible PyTorch and Torchvision build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### Dataset Setup

The datasets are not included in the repository. Place the So2Sat files in `data/raw/`:

```text
data/
└── raw/
    ├── validation.h5
    └── testing.h5
```

The saved So2Sat split indices belong in:

```text
configs/so2sat_split.npz
```

EuroSAT must also be available locally. Configure its dataset directory according to the path expected by `src/training/train_eurosat.py`.

### Training

Run commands from the repository root.

First, train the EuroSAT model:

```bash
python -m src.training.train_eurosat
```

Then train the So2Sat configurations:

**M1: Random initialization, full fine-tuning**

```bash
python -m src.training.train_so2sat --experiment M1 --validation-h5 data/raw/validation.h5 --split-path configs/so2sat_split.npz --output-dir checkpoints/M1
```

**M2: ImageNet initialization, frozen backbone**

```bash
python -m src.training.train_so2sat --experiment M2 --validation-h5 data/raw/validation.h5 --split-path configs/so2sat_split.npz --output-dir checkpoints/M2
```

**M3: ImageNet → EuroSAT, frozen backbone**

```bash
python -m src.training.train_so2sat --experiment M3 --validation-h5 data/raw/validation.h5 --split-path configs/so2sat_split.npz --eurosat-checkpoint checkpoints/eurosat/best_checkpoint.pth --output-dir checkpoints/M3
```

**M4: ImageNet → EuroSAT, full fine-tuning**

```bash
python -m src.training.train_so2sat --experiment M4 --validation-h5 data/raw/validation.h5 --split-path configs/so2sat_split.npz --eurosat-checkpoint checkpoints/eurosat/best_checkpoint.pth --output-dir checkpoints/M4
```

**M5: ImageNet initialization, full fine-tuning**

```bash
python -m src.training.train_so2sat --experiment M5 --validation-h5 data/raw/validation.h5 --split-path configs/so2sat_split.npz --output-dir checkpoints/M5
```

The scripts support additional arguments for epochs, batch size, and random seed.

### Evaluation

Evaluate the saved checkpoints on the official testing partition:

```bash
python -m src.evaluation.evaluate_so2sat --test-h5 data/raw/testing.h5 --checkpoint-root checkpoints --output-root results/so2sat
```

The evaluation script generates comparative metrics, classification reports, confusion matrices, and prediction files.

The official testing partition must remain separate from training, hyperparameter tuning, and checkpoint selection.

### Checkpoints and Results

Each experiment saves its best checkpoint, selected using internal validation loss, along with training history and evaluation outputs.

The EuroSAT checkpoint is required for M3 and M4. M1, M2, and M5 do not require it.

Large model checkpoints and raw datasets should not be committed directly to Git. If distributed, checkpoints should be provided through GitHub Releases or another file-hosting service.

### Reproducibility Considerations

The reported experiments use random seed 42 and one training/validation split. Re-running experiments may produce small differences due to hardware, library versions, and nondeterministic GPU operations.

Reproducing the exact reported metrics requires the original checkpoints and corresponding dataset partitions.
