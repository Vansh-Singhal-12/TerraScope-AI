# TerraScope AI: Experimental Results

This directory contains the training and evaluation artifacts for the five ResNet-18 experiments conducted on So2Sat LCZ42.

The experiments investigate transfer learning, additional satellite-specific pretraining, and fine-tuning.

## Experimental Configurations

| Model | Initialization | So2Sat training |
|---|---|---|
| M1 | Random | Full fine-tuning |
| M2 | ImageNet | Frozen backbone |
| M3 | ImageNet -> EuroSAT | Frozen backbone |
| M4 | ImageNet -> EuroSAT | Full fine-tuning |
| M5 | ImageNet | Full fine-tuning |

## Validation Results

| Model | Validation Loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| M1 | 1.330417 | 68.99% | 59.50% | 68.99% |
| M2 | 1.235399 | 59.02% | 43.46% | 57.01% |
| M3 | 1.437830 | 51.14% | 32.92% | 46.72% |
| M4 | 0.295866 | 92.04% | 89.35% | 91.98% |
| M5 | 0.301955 | 92.23% | 89.45% | 92.16% |

## Experimental Interpretation

### 1. Transfer Learning Compared with Random Initialization

M1 provides the random-initialization baseline.

M4 and M5 substantially outperform M1 on the validation partition, providing preliminary evidence that pretrained representations are useful for the target classification task.

These comparisons also involve differences in training dynamics, so they should not be interpreted as perfectly isolated measurements of pretraining alone.

### 2. Effect of Additional EuroSAT Pretraining

M2 and M3 both use frozen feature extractors.

- M2 uses ImageNet initialization.
- M3 uses ImageNet initialization followed by EuroSAT pretraining.

M2 achieves 59.02% accuracy, compared with 51.14% for M3.

Under this frozen-feature configuration, additional EuroSAT pretraining does not improve validation performance.

### 3. Effect of Fine-Tuning

For the ImageNet-to-EuroSAT pathway:

- M3: 51.14% accuracy
- M4: 92.04% accuracy

For the ImageNet pathway:

- M2: 59.02% accuracy
- M5: 92.23% accuracy

Both pathways benefit substantially from full fine-tuning.

### 4. Comparison of Pretraining Pathways

M4 and M5 are the primary comparison for evaluating the effect of additional EuroSAT pretraining when both models are fully fine-tuned.

- M4 accuracy: 92.04%
- M5 accuracy: 92.23%
- Difference: approximately 0.19 percentage points in favor of M5.

M4 has a slightly lower validation loss, while M5 has slightly higher accuracy, macro-F1, and weighted-F1.

The difference is very small. These results do not establish a meaningful performance advantage for either pathway.

## EuroSAT Pretraining

The EuroSAT pretraining experiment achieved:

- Validation accuracy: 97.09%
- Macro-F1: 96.99%
- Weighted-F1: 97.09%

These metrics describe performance on EuroSAT and should not be directly compared with So2Sat validation metrics because the datasets have different label spaces and distributions.

## Artifact Organization

Each So2Sat experiment contains four compact result files:

| File | Description |
|---|---|
| validation_metrics.json | Overall validation metrics |
| classification_report.csv | Per-class precision, recall, F1-score, and support |
| confusion_matrix.csv | Confusion matrix |
| training_history.json | Epoch-level training and validation history |

Model checkpoints are stored separately from GitHub because of their size.

## Evaluation Protocol

- Dataset: So2Sat LCZ42
- Internal training samples: 19,295
- Internal validation samples: 4,824
- Official test samples: 24,188
- Number of classes: 17
- Random seed: 42
- Checkpoint selection: Lowest validation loss

The official test partition is reserved for final evaluation and is not used for model selection or hyperparameter tuning.

## Current Status

- [x] M1 training and validation
- [x] M2 training and validation
- [x] M3 training and validation
- [x] M4 training and validation
- [x] M5 training and validation
- [x] Preliminary comparisons
- [ ] Official test-set evaluation
- [ ] Final comparative analysis
- [ ] Final figures and visualizations
