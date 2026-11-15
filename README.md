# TerraScope AI

## Satellite Image Classification Through Sequential Transfer Learning

## Overview

TerraScope AI is a satellite-image classification project that applies sequential transfer learning to identify Local Climate Zones (LCZs). I developed a ResNet-18 model that classifies satellite imagery into 17 categories representing different urban forms and land-cover types.

My approach adapts representations learned from general-purpose image data and satellite-specific imagery to a more specialized classification task. I begin with ImageNet-pretrained weights, further pretrain the model on EuroSAT, and then fine-tune it on So2Sat LCZ42. This process produces M4, my primary demonstration model.

Alongside developing the classifier, I investigate how different pretraining pathways and fine-tuning strategies affect performance. I also examine the geographic distribution of training and evaluation samples to better understand the limitations of the model's apparent generalization.

My longer-term goal is to develop models that can adapt reliably to satellite imagery from geographic regions that were not represented during training.

## Research Question

To what extent can sequential transfer learning across general-purpose and satellite-specific datasets produce an effective Local Climate Zone classifier, and how do pretraining pathways and fine-tuning strategies influence its performance and geographic generalization?

## Objectives

- Develop a 17-class Local Climate Zone classifier using Sentinel-2 satellite imagery from So2Sat LCZ42.
- Apply sequential transfer learning from ImageNet to EuroSAT and then to So2Sat.
- Compare random initialization, ImageNet initialization, and ImageNet-to-EuroSAT pretraining.
- Evaluate the effect of freezing the pretrained backbone versus fully fine-tuning the model.
- Compare model performance using accuracy, macro-F1, weighted-F1, and loss.
- Examine the geographic distribution of training and evaluation samples to identify potential limitations in generalization.

## Dataset

### So2Sat LCZ42

The primary dataset is So2Sat LCZ42, which contains Sentinel-2 satellite imagery labeled with 17 Local Climate Zone categories.

Local Climate Zones describe urban and natural environments according to characteristics such as building density, building height, surface cover, and vegetation. The 17 categories include ten built types and seven land-cover types.

The dataset provides 32 × 32 satellite-image patches. For this project, I use the first three Sentinel-2 bands in RGB order corresponding to B4, B3, and B2.

| Partition | Samples | Purpose |
|---|---:|---|
| Official validation partition | 24,119 | Training and internal validation |
| Training subset | 19,295 | Model training |
| Internal validation subset | 4,824 | Model selection |
| Official test partition | 24,188 | Final evaluation |

I divided the official validation partition into training and internal validation subsets using an 80/20 split with random seed 42. The separate official test partition was reserved for final evaluation.

### EuroSAT

EuroSAT is used as an intermediate satellite-specific pretraining dataset. It contains satellite images across ten land-use and land-cover classes.

Using EuroSAT allows the model to learn satellite-image representations before being adapted to the more specialized So2Sat LCZ42 classification task.

### ImageNet

ImageNet provides the initial pretrained weights for the ResNet-18 architecture. These weights give the model general visual representations before satellite-specific training begins.

## Model Architecture and Preprocessing

I use ResNet-18, a convolutional neural network architecture, as the base classifier.

I adapted the original architecture for the satellite imagery used in this project:

- The input consists of 32 × 32 RGB satellite-image patches.
- The initial convolutional layer uses a stride of (1, 1).
- The original max-pooling layer is removed to preserve spatial information in the relatively small input images.
- The final fully connected layer is replaced with a 17-class classification head.

### Image preprocessing

The satellite reflectance values are clipped to the range 0–2.8 and divided by 2.8 to scale them to approximately 0–1. The resulting images are then normalized using ImageNet channel means and standard deviations.

This preprocessing is applied consistently during model training and evaluation.

## Transfer-Learning Experiments

I compare five model configurations to investigate the effects of initialization, intermediate satellite-specific pretraining, and fine-tuning.

| Model | Initialization | So2Sat training strategy |
|---|---|---|
| M1 | Random initialization | Full fine-tuning |
| M2 | ImageNet | Frozen backbone |
| M3 | ImageNet → EuroSAT | Frozen backbone |
| M4 | ImageNet → EuroSAT | Full fine-tuning |
| M5 | ImageNet | Full fine-tuning |

M3 and M4 use the same EuroSAT-pretrained checkpoint. Their difference is whether the backbone remains frozen or is fully fine-tuned during So2Sat training.

M2 and M5 provide the corresponding comparison for ImageNet initialization.

### Primary model: M4

M4 is my primary demonstration model. It follows the sequential transfer-learning pathway:

**ImageNet → EuroSAT → So2Sat LCZ42**

The model first learns general visual representations from ImageNet, then develops satellite-specific representations through EuroSAT, and finally adapts to the 17-class Local Climate Zone task through full fine-tuning on So2Sat.

I selected M4 as the practical model developed through this pipeline. It achieved the lowest internal validation loss among the fully fine-tuned models initialized with pretrained weights. Its official test performance was also comparable to M5.

The results do not establish that the ImageNet-to-EuroSAT pathway is definitively superior to ImageNet-only initialization. Instead, they demonstrate that sequential transfer learning can produce a strong classifier while highlighting the importance of fine-tuning strategy.

## Training Configuration

The So2Sat experiments use the following training configuration:

| Setting | Value |
|---|---|
| Architecture | ResNet-18 |
| Number of classes | 17 |
| Input size | 32 × 32 × 3 |
| Epochs | 12 |
| Batch size | 64 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Backbone learning rate | 0.0001 |
| Classification head learning rate | 0.001 |
| Learning-rate scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 2 |
| Random seed | 42 |
| Precision | Automatic mixed precision on CUDA |

For frozen-backbone experiments, I keep the backbone in evaluation mode to prevent its BatchNorm running statistics from changing during training.

I retain the checkpoint with the lowest internal validation loss for each experiment.

## Results

### Internal validation results

The following results were obtained on the internal validation subset of 4,824 samples.

| Model | Validation loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| M1 | 1.3304 | 68.99% | 59.50% | 68.99% |
| M2 | 1.2354 | 59.02% | 43.46% | 57.01% |
| M3 | 1.4378 | 51.14% | 32.92% | 46.72% |
| M4 | 0.2959 | 92.04% | 89.35% | 91.98% |
| M5 | 0.3020 | 92.23% | 89.45% | 92.16% |

### Official test results

The final evaluation uses the separate official test partition containing 24,188 samples.

| Model | Test loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| M1 | 1.9156 | 60.76% | 0.4922 | 0.6024 |
| M2 | 1.3007 | 57.69% | 0.4276 | 0.5557 |
| M3 | 1.5030 | 50.01% | 0.3236 | 0.4557 |
| M4 | 1.2607 | 77.14% | 0.6981 | 0.7658 |
| M5 | 1.3104 | 76.81% | 0.6995 | 0.7637 |

### Interpretation

Several findings emerge from the experiments:

1. **Full fine-tuning substantially improves performance.**

   M3 achieved 50.01% test accuracy with a frozen backbone, while M4 achieved 77.14% after full fine-tuning. Similarly, M5 outperformed M2 by approximately 19.12 percentage points. These comparisons indicate that allowing the pretrained backbone to adapt to the target dataset was important for this task.

2. **Intermediate EuroSAT pretraining did not produce a clear performance advantage.**

   M4 and M5 achieved very similar test results. M4 had slightly higher accuracy and weighted-F1, while M5 had slightly higher macro-F1. The differences are small and do not establish a definitive winner between the two initialization pathways.

3. **Pretraining alone was not sufficient.**

   The frozen-backbone models performed substantially worse than their fully fine-tuned counterparts. This suggests that general-purpose and satellite-specific representations still require adaptation to the target LCZ classification task.

4. **The validation-to-test performance gap requires further investigation.**

   M4 achieved 92.04% internal validation accuracy but 77.14% on the official test partition. This difference motivated a separate investigation into the geographic structure of the dataset.

## Geographic Generalization Analysis

A central challenge in satellite-image classification is determining whether a model has learned transferable environmental patterns or has benefited from geographic similarities between its training and evaluation samples.

To investigate this, I analyzed the geographic coordinates associated with the So2Sat samples.

### Dataset composition

A chi-square test comparing the class distributions of the internal validation and official test partitions produced:

- Chi-square statistic: 14.396
- Degrees of freedom: 16
- p-value: 0.5692

This analysis did not detect a statistically significant difference in class composition between the two partitions.

I also examined the geographic distribution. Both partitions contain samples from the same ten cities, with each city contributing approximately 10% of the samples.

### Geographic distance

I calculated the distance from each evaluation sample's center to its nearest training-sample center.

| Measurement | Internal validation | Official test |
|---|---:|---:|
| Median nearest-center distance | 100 m | 2,342 m |
| Mean nearest-center distance | 146.32 m | 3,585.98 m |
| 90th percentile | 223.61 m | 8,224.17 m |

Each image patch covers approximately 320 m × 320 m, based on its 32 × 32 pixels at 10 m per pixel.

I also examined whether the geographic footprints of evaluation patches intersected the footprints of training patches.

| Partition | Patches with positive-area footprint intersection |
|---|---:|
| Internal validation | 4,713 / 4,824 (97.70%) |
| Official test | 2,002 / 24,188 (8.28%) |

### Interpretation and limitations

The internal validation samples were geographically much closer to the training samples than the official test samples. The difference in geographic proximity, alongside the validation-to-test performance gap, suggests that spatial dependence may contribute to the difference in model performance.

However, this analysis does not establish geographic overlap as the cause of the performance gap. A positive-area footprint intersection also does not prove that two patches contain identical pixels.

Most importantly, both the internal validation and official test partitions contain samples from the same ten cities. Therefore, the official test results do not demonstrate generalization to entirely unseen cities or geographic regions.

The spatial analysis identifies an important limitation and motivates a more rigorous future evaluation using geographically separated training and testing regions.

For the complete analysis, see docs/generalization_analysis.md.

## Conclusion

TerraScope AI demonstrates an applied transfer-learning pipeline for Local Climate Zone classification using satellite imagery.

By adapting ResNet-18 from ImageNet to EuroSAT and then to So2Sat LCZ42, I developed M4, a 17-class classifier that achieved 77.14% accuracy on the official test partition. Comparisons across five configurations show that full fine-tuning was important to performance, while the additional benefit of EuroSAT pretraining over ImageNet-only initialization remains inconclusive.

My geographic analysis also revealed a substantial difference in the spatial proximity of internal validation and official test samples. This finding highlights why strong performance on a conventional held-out partition should not automatically be interpreted as geographic generalization.

This project establishes a foundation for further work on adapting satellite-image classifiers across datasets and geographic regions.

## Future Work

Future development will focus on:

- **Unseen-region evaluation:** Train on selected cities and test on entirely different cities to directly measure geographic generalization.
- **Repeated experiments:** Run multiple random seeds to determine whether the observed performance differences are consistent.
- **Class-level analysis:** Investigate confusion patterns and improve performance on underrepresented LCZ categories.
- **Transfer-learning optimization:** Explore alternative pretraining strategies, fine-tuning schedules, and model architectures.
- **Regional adaptation:** Investigate how the model can be adapted to satellite imagery from new geographic areas with limited labeled data.

## Repository Structure

```text
TerraScope-AI/
├── configs/
│   ├── experiment.json
│   └── so2sat_split.npz
├── data/
│   └── raw/
│       ├── validation.h5
│       └── testing.h5
├── docs/
│   └── generalization_analysis.md
├── results/
│   └── so2sat/
├── src/
│   ├── analysis/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── training/
├── checkpoints/
├── requirements.txt
└── README.md
```

Raw datasets and model checkpoints are not included in the Git repository. They must be obtained separately.

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/Vansh-Singhal-12/TerraScope-AI.git
cd TerraScope-AI
```

### 2. Install dependencies

The project dependencies and versions are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

The experiments were developed using PyTorch with CUDA acceleration in Google Colab.

### 3. Prepare the datasets

Place the So2Sat files in the following locations:

```text
data/raw/validation.h5
data/raw/testing.h5
```

The project uses the official validation and testing partitions. The internal training/validation split is stored in:

```text
configs/so2sat_split.npz
```

The EuroSAT dataset must also be made available for intermediate pretraining.

### 4. Train the EuroSAT model

The EuroSAT training script produces the checkpoint used by the sequential transfer-learning experiments.

```bash
python -m src.training.train_eurosat \
    --data-root ./data \
    --checkpoint-dir ./checkpoints \
    --epochs 12 \
    --batch-size 64 \
    --seed 42
```

The resulting checkpoint is:

```text
checkpoints/eurosat_resnet18_best.pth
```

### 5. Train the So2Sat models

The So2Sat training script is located at:

```text
src/training/train_so2sat.py
```

It supports the five experimental configurations described above. M3 and M4 require the EuroSAT-pretrained checkpoint.

To view the available command-line arguments:

```bash
python -m src.training.train_so2sat --help
```

The experiment outputs are organized under:

```text
results/so2sat/
```

### 6. Evaluate the models

After training, the models can be evaluated on the official test partition:

```bash
python -m src.evaluation.evaluate_so2sat \
    --test-h5 data/raw/testing.h5 \
    --checkpoint-root results/so2sat \
    --output-root results/so2sat \
    --experiments M1 M2 M3 M4 M5 \
    --batch-size 64
```

The evaluation script produces the metrics and outputs used in the results section.

## Sources and Acknowledgments

### Datasets and pretrained models

- **So2Sat LCZ42:** Dataset used for 17-class Local Climate Zone classification.
- **EuroSAT:** Satellite-image dataset used for intermediate pretraining.
- **ImageNet:** Source of the initial pretrained ResNet-18 weights.
- **ResNet:** He et al., Deep Residual Learning for Image Recognition (2015).

### Software and tools

- **PyTorch:** Model implementation and training.
- **Torchvision:** ResNet-18 architecture and pretrained weights.
- **NumPy:** Numerical processing.
- **h5py:** HDF5 dataset access.
- **scikit-learn:** Classification metrics and evaluation.
- **SciPy:** Statistical analysis.
- **Matplotlib:** Data visualization.
- **Google Colab:** Development and GPU-assisted experimentation.
- **GitHub:** Version control and project hosting.

I acknowledge the original creators of all datasets, pretrained weights, libraries, and external resources used in this project. Their sources should be cited in the project documentation.