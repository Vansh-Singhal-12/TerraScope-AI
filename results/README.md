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


## Official Test Results

The five models were evaluated on the official So2Sat testing partition,
containing 24,188 samples. The test partition was held out from model
selection.

| Model | Test loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| M1 | 1.9156 | 60.76% | 49.22% | 60.24% |
| M2 | 1.3007 | 57.69% | 42.76% | 55.57% |
| M3 | 1.5030 | 50.01% | 32.36% | 45.57% |
| M4 | 1.2607 | 77.14% | 69.81% | 76.58% |
| M5 | 1.3104 | 76.81% | 69.95% | 76.37% |

### Per-model files

Each `results/so2sat/Mn/` directory contains:

- `test_metrics.json`: aggregate official test metrics
- `test_classification_report.csv`: per-class precision, recall, and F1
- `test_confusion_matrix.csv`: confusion matrix
- `test_predictions.npz`: saved test predictions and associated arrays

The combined comparison is stored in:

- `official_test_comparison.csv`

## Geographic Distribution Audit

The `results/so2sat/distribution_audit/` directory contains the outputs
of the geographic-generalization investigation.

Important files include:

| File | Contents |
|---|---|
| `class_distribution.csv` | Internal-validation and test class distributions |
| `city_distribution.csv` | City representation across partitions |
| `class_by_city_difference.csv` | Class-composition differences by city |
| `validation_vs_test_per_city.csv` | Per-city validation and test comparison |
| `per_city_test_metrics.csv` | Per-city test metrics |
| `per_city_test_metrics_corrected.csv` | Corrected per-city test metrics |
| `nearest_training_distance_summary.csv` | Summary of distances to nearest training samples |
| `nearest_training_distances.csv` | Individual nearest-training-sample distances |
| `training_footprint_overlap_overall.csv` | Overall training-footprint overlap |
| `training_footprint_overlap_by_city.csv` | Footprint overlap broken down by city |
| `spatial_distribution_by_city.png` | Spatial distribution visualization |

### Main spatial findings

| Measurement | Internal validation | Official test |
|---|---:|---:|
| Samples | 4,824 | 24,188 |
| Median nearest-training-center distance | 100.00 m | 2,342.01 m |
| Samples intersecting training footprints | 97.70% | 8.28% |

Footprint intersection indicates geographic overlap between patch
rectangles. It does not establish identical pixels or prove that spatial
overlap caused the performance difference.

The full methodology, interpretation, and limitations are documented in
[`docs/generalization_analysis.md`](../docs/generalization_analysis.md).

## Interpretation Notes

- M4 and M5 have nearly identical official test performance; the small
  metric differences do not establish a definitive pretraining-pathway winner.
- The internal validation split is a random sample-level split and exhibits
  substantial spatial association with training samples.
- The official test partition is geographically more separated, but still
  represents the same ten cities as the training data.
- The EuroSAT checkpoints used for M3 and M4 were initialized with
  ImageNet-pretrained weights before EuroSAT pretraining.
