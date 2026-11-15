# Geographic Generalization and Spatial Dependence

## Purpose

The internal validation results were substantially higher than the official
So2Sat test results for the fully fine-tuned models. This investigation
examined whether differences in geographic proximity between training,
internal validation, and official test samples could help explain the gap.

The analysis considered class distributions, city distributions, per-city
performance, nearest-training-sample distances, and image-footprint overlap.

## Dataset partitions

The official So2Sat validation partition contains 24,119 samples. It was
divided into:

- Internal training: 19,295 samples
- Internal validation: 4,824 samples

The official testing partition contains 24,188 samples.

Both official partitions cover the same ten named cities, but their
geographic samples are distributed across separated areas. The experiment
therefore evaluates geographic separation within represented cities, not
generalization to entirely unseen cities.

## Validation-to-test performance gap

| Model | Validation accuracy | Test accuracy | Change |
|---|---:|---:|---:|
| M1 | 68.99% | 60.76% | -8.23 pp |
| M2 | 59.02% | 57.69% | -1.33 pp |
| M3 | 51.14% | 50.01% | -1.13 pp |
| M4 | 92.04% | 77.14% | -14.90 pp |
| M5 | 92.23% | 76.81% | -15.42 pp |

The fully fine-tuned models show the largest validation-to-test accuracy
decline. M4 and M5 also experience substantial declines in macro-F1.

### Official test performance

| Model | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| M4 | 77.14% | 69.81% | 76.58% |
| M5 | 76.81% | 69.95% | 76.37% |

Although M4 and M5 achieved validation accuracies above 92%, their official
test accuracies were approximately 77%. Their test macro-F1 scores were
also substantially lower than their validation macro-F1 scores, indicating
that the decline was not limited to the most frequently represented
classes.

## Class and city distributions

The internal validation and official test class distributions were
compared using a chi-square test.

- Chi-square statistic: 14.396
- Degrees of freedom: 16
- p-value: 0.569

The test did not detect a statistically significant difference in the
overall class distributions.

City proportions were also similar, with each city accounting for
approximately one tenth of each partition.

These comparisons suggest that the performance gap is not readily explained
by a major difference in the overall class or city proportions. They do not
rule out differences in local class composition or image characteristics.

## Per-city performance

M4 and M5 had lower test accuracy than internal-validation accuracy in
all ten cities.

For M4, the per-city accuracy decline ranged from approximately 5.08
percentage points in Moscow to 29.01 points in Nairobi.

For M5, it ranged from approximately 5.81 points in Moscow to 27.59
points in Nairobi.

The decline across all cities is consistent with a broader geographic
generalization challenge rather than a problem isolated to one city.

## Distance to nearest training sample

For each internal-validation and official-test patch, the distance to its
nearest training-patch center was calculated within the same city using
projected coordinates.

| Metric | Internal validation | Official test |
|---|---:|---:|
| Median distance | 100.00 m | 2,342.01 m |
| Mean distance | 146.32 m | 3,585.98 m |
| 90th percentile | 223.61 m | 8,224.17 m |
| Within 250 m | 93.39% | 4.94% |
| Within 500 m | 98.61% | 11.54% |
| Within 1 km | 99.48% | 24.03% |

The internal-validation samples were substantially closer to training
samples than official-test samples.

## Image-footprint overlap

Each input patch contains 32 × 32 pixels at 10 m per pixel, corresponding
to an approximately 320 m × 320 m ground footprint.

For each validation or test patch, the analysis checked whether its
axis-aligned rectangular footprint had positive-area intersection with
at least one training-patch footprint in the same projected coordinate
system.

| Partition | Samples | Samples overlapping training | Percentage |
|---|---:|---:|---:|
| Internal validation | 4,824 | 4,713 | 97.70% |
| Official test | 24,188 | 2,002 | 8.28% |

The difference is substantial. Nearly all internal-validation patches
intersected at least one training-patch footprint, whereas this occurred
for a much smaller fraction of official-test patches.

This establishes geographic footprint intersection, not necessarily
identical pixel values or identical predictions. The analysis does not
measure the causal contribution of overlap to model performance.

## Interpretation

The combined findings indicate that the random internal validation split
provided an evaluation setting with substantial spatial dependence between
training and validation imagery.

The official test samples were generally much farther from training
samples and had substantially less footprint overlap. This provides a
plausible explanation for why internal-validation performance was
optimistic relative to official-test performance.

The results support the importance of spatially informed evaluation for
satellite-image classification. Random sample-level splits may not
adequately estimate performance on geographically separated imagery
when nearby patches share spatial context.

The analysis does not prove that spatial dependence is the sole cause of
the performance gap. Differences in local scene composition, image
characteristics, and other forms of distribution shift may also contribute.

## Limitations

- The ten cities are represented in both partitions; this is not an
  unseen-city evaluation.
- The internal validation split uses one random seed and one partition.
- Footprint intersection does not establish identical input pixels.
- The nearest-neighbor calculation uses patch-center distances.
- The overlap calculation assumes axis-aligned 320 m square footprints.
- The investigation is observational and does not isolate the causal
  effect of spatial overlap.
- The study does not include repeated spatial-block cross-validation.

## Reproducibility

The numerical outputs are stored in:

`results/so2sat/distribution_audit/`

Important artifacts include:

- `class_distribution.csv`
- `city_distribution.csv`
- `class_by_city_difference.csv`
- `validation_vs_test_per_city.csv`
- `nearest_training_distance_summary.csv`
- `nearest_training_distances.csv`
- `training_footprint_overlap_by_city.csv`
- `training_footprint_overlap_overall.csv`
- `spatial_distribution_by_city.png`

The official test metrics and predictions are stored in the individual
model result directories under `results/so2sat/`.
