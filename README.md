
# TerraScope AI

**Investigating Geographic Transfer Learning in Satellite Image Classification**

## Overview

TerraScope AI investigates whether learning land-cover representations from satellite imagery in one geographic region improves classification performance in geographically different regions.

The project compares models trained from scratch, models using a frozen pretrained feature extractor, and models fine-tuned on the target dataset.

## Research Question

Does pretraining on EuroSAT improve performance on a geographically different dataset, and how much additional benefit does fine-tuning provide?

## Datasets

- **EuroSAT:** Sentinel-2 satellite imagery with 10 land-use and land-cover classes.
- **So2Sat LCZ42:** Sentinel-1 and Sentinel-2 imagery labeled across 17 Local Climate Zones (LCZs).

So2Sat LCZ42 is used as the target classification task.

## Experimental Design

Three approaches will be compared:

1. **Target-only baseline:** A ResNet-18 trained on So2Sat without EuroSAT pretraining.
2. **Frozen transfer:** A ResNet-18 pretrained on EuroSAT, with its feature extractor frozen while a new So2Sat classification head is trained.
3. **Fine-tuned transfer:** The EuroSAT-pretrained model further trained on So2Sat.

All three approaches will be evaluated using the same held-out target test partition.

## Evaluation

The evaluation will include:

- Accuracy
- Macro-F1 and weighted-F1
- Per-class precision and recall
- Confusion matrices
- Comparison of transfer-learning approaches

## Technology

- Python
- PyTorch
- torchvision
- NumPy
- h5py
- scikit-learn
- Matplotlib

## Reproducibility

Code, configurations, methodology, and experiment results will be maintained in this repository.

Large datasets and model checkpoints will be stored separately rather than committed to GitHub.

## Project Status

Currently in development. Dataset exploration and preprocessing are underway.

## Limitations

The target dataset's official validation partition is being considered for model adaptation, while its official testing partition will be reserved for final evaluation. This differs from the conventional training/validation/test workflow and will be documented in the final methodology.
