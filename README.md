# TerraScope AI
### Satellite Image Classification Through Transfer Learning

## Overview

TerraScope AI is a deep-learning project that classifies satellite imagery into 17 Local Climate Zone (LCZ) categories using Sentinel-2 data.

Beyond developing an accurate image classifier, this project investigates how transfer learning affects satellite-image classification. In particular, we examine whether pretrained visual representations improve classification, whether additional satellite-specific pretraining provides benefits, and how fine-tuning influences transfer performance.

We compare models initialized randomly, pretrained on ImageNet, and pretrained sequentially on ImageNet and EuroSAT before adapting them to So2Sat.

## Research Question

**To what extent does transfer learning improve satellite-image classification, and how does the choice of pretraining pathway influence its effectiveness across geographically distinct datasets?**

### Research Objectives

1. Evaluate the effect of transfer learning compared with training a model from random initialization.
2. Investigate whether additional satellite-specific pretraining on EuroSAT improves transfer performance over ImageNet initialization alone.
3. Examine how freezing or fine-tuning pretrained feature extractors affects classification performance.
4. Evaluate whether these differences persist when models are tested on held-out So2Sat data.

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

M4 versus M5 is the primary comparison for investigating whether additional satellite-specific pretraining provides an advantage when both models are fully adapted to So2Sat.

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

The official test evaluation will determine whether these observations persist on held-out samples.

## Geographic Transfer and Limitations

EuroSAT and So2Sat represent different datasets and geographic contexts, allowing us to investigate cross-dataset transfer.

However, the official So2Sat validation and testing partitions cover the same ten cities. Consequently, this experiment does not establish generalization to entirely unseen cities.

Other limitations include:

- A single random seed and training/validation split.
- A limited set of pretraining pathways and model configurations.
- Class imbalance across the 17 LCZ categories.
- The use of RGB bands rather than the full Sentinel-2 spectral information.

These limitations are considered when interpreting the results.
