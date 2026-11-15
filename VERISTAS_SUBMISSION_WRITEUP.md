# TerraScope AI: Satellite-Based Local Climate Zone Classification

**Track:** Climate  
**Project:** VERITAS Project 4.3.3, Land-Use Mapping from Satellite Images

**Project Author:**
- Vansh Singhal | vansh12singhal@gmail.com | Terra Environmental Research Institute

## Problem

Satellite imagery provides a way to study how land is used and how urban environments are structured. However, classifying these environments requires models that can recognize meaningful patterns across different geographic areas.

I developed TerraScope AI, a satellite-image classification pipeline that uses transfer learning to identify 17 Local Climate Zone (LCZ) categories. These categories represent different built environments and natural land-cover types.

## What I Built

I developed a ResNet-18 classifier trained to recognize 17 Local Climate Zone categories using Sentinel-2 imagery from the So2Sat LCZ42 dataset.

The model processes 32 × 32 RGB satellite-image patches and predicts one of the 17 LCZ categories. I implemented a sequential transfer-learning pipeline:

**ImageNet → EuroSAT → So2Sat LCZ42**

This allows the model to begin with general visual representations, learn satellite-specific features, and then adapt to the target LCZ classification task.

Beyond classification accuracy, TerraScope AI has practical applications in urban climate research and environmental planning. Local Climate Zone predictions provide a baseline for climate scientists and city planners studying urban heat islands, tracking sprawl, and mapping structural differences between built and natural environments. Because the pipeline runs on Sentinel-2 imagery, it demonstrates how openly available Earth-observation data can support scalable environmental monitoring. With further validation across geographically unseen regions, this framework could directly inform climate-risk assessments and disaster preparedness.

## What I Added

VERITAS Project 4.3.3 proposed building a satellite-image classifier for broad land-use categories such as forests, farmland, rivers, and towns. Its suggested starting point was EuroSAT, a dataset containing ten land-use and land-cover classes, with a pretrained ResNet-18 as a potential model architecture.

I extended the project beyond that baseline in three main ways:

- **More specialized classification:** Rather than stopping at EuroSAT's ten broad categories, I incorporated So2Sat LCZ42 to classify 17 Local Climate Zones, distinguishing between different built environments and natural land-cover types.
- **A five-model ablation study:** I designed five configurations to compare random initialization, ImageNet initialization, and sequential ImageNet-to-EuroSAT pretraining, while also examining the effect of freezing versus fully fine-tuning the backbone.
- **Spatial generalization audit:** Beyond standard accuracy evaluation, I examined geographic distributions, distances between sample centers, and spatial footprint overlap between training, validation, and test patches. This helped investigate the substantial difference between internal validation and official test performance.

These extensions shifted the project from a basic land-use classification task toward an investigation of transfer learning and geographic generalization in satellite imagery.

## How It Works

The primary dataset, So2Sat LCZ42, contains 17 LCZ categories represented by 32 × 32 Sentinel-2 image patches. I use the B4, B3, and B2 bands as RGB inputs.

The official validation partition contains 24,119 samples. I divided it into 19,295 training samples and 4,824 internal validation samples using an 80/20 split with random seed 42. The separate official testing partition contains 24,188 samples and was reserved for final evaluation.

I compared five model configurations:

| Model | Initialization | So2Sat Training |
|---|---|---|
| M1 | Random | Full fine-tuning |
| M2 | ImageNet | Frozen backbone |
| M3 | ImageNet → EuroSAT | Frozen backbone |
| M4 | ImageNet → EuroSAT | Full fine-tuning |
| M5 | ImageNet | Full fine-tuning |

## Results

The primary demonstration model, M4, achieved **77.14% accuracy** and **0.6981 macro-F1** on the official test partition.

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| M1 | 60.76% | 0.4922 |
| M2 | 57.69% | 0.4276 |
| M3 | 50.01% | 0.3236 |
| M4 | 77.14% | 0.6981 |
| M5 | 76.81% | 0.6995 |

The experiments indicate that full fine-tuning was important for performance. M4 and M5 performed very similarly: M4 achieved marginally higher accuracy, while M5 achieved marginally higher macro-F1. Their results do not establish that intermediate EuroSAT pretraining is definitively better than ImageNet-only initialization. Instead, they demonstrate that both initialization pathways can achieve comparable results when followed by full fine-tuning.

One important finding was the difference between internal validation and official test performance. M4 achieved 92.04% validation accuracy but 77.14% test accuracy.

My geographic analysis found that internal validation samples were much closer to training samples than official test samples. Approximately 97.70% of internal validation patches had positive-area footprint intersections with training patches, compared with 8.28% of official test patches.

Both partitions contain samples from the same ten cities, however, so these experiments do not establish generalization to entirely unseen cities. The spatial findings suggest a possible contribution from geographic dependence, but do not prove that it caused the performance gap.

## Next Steps

I plan to evaluate the model on geographically held-out cities or regions, repeat experiments across multiple random seeds, and investigate class-level errors. These experiments would provide a stronger assessment of whether the classifier can adapt reliably to satellite imagery from locations not represented during training.

## Sources and Acknowledgments

### Datasets and Pretrained Models

- **So2Sat LCZ42:** Zhu et al. (2020). *So2Sat LCZ42: A Benchmark Dataset for the Classification of Global Local Climate Zones*. [Dataset repository](https://github.com/zhu-xlab/So2Sat-LCZ42) | [Original dataset record](https://mediatum.ub.tum.de/1483140) | [Research publication](https://doi.org/10.1109/MGRS.2020.2964708)
- **EuroSAT:** Helber et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. [Dataset repository](https://github.com/phelber/eurosat) | [Research publication](https://doi.org/10.1109/JSTARS.2019.2918242)
- **ImageNet:** [Official dataset website](https://www.image-net.org/)
- **ResNet:** He et al. (2015). *Deep Residual Learning for Image Recognition*. [Original paper](https://arxiv.org/abs/1512.03385)

### Software and Libraries

- **PyTorch:** Model implementation and training. [Official documentation](https://docs.pytorch.org/docs/stable/index.html)
- **Torchvision:** ResNet-18 architecture and pretrained weights. [Official documentation](https://docs.pytorch.org/vision/stable/index.html)
- **NumPy:** Numerical processing. [Official documentation](https://numpy.org/doc/stable/)
- **h5py:** HDF5 dataset access. [Official documentation](https://docs.h5py.org/en/stable/)
- **scikit-learn:** Classification metrics and evaluation. [Official documentation](https://scikit-learn.org/stable/)
- **SciPy:** Statistical analysis. [Official documentation](https://docs.scipy.org/doc/scipy/)
- **Matplotlib:** Data visualization. [Official documentation](https://matplotlib.org/stable/)

### Development Tools

- **Google Colab:** Development and GPU-assisted experimentation. [Official website](https://colab.research.google.com/)
- **GitHub:** Version control and project hosting. [Project repository](https://github.com/Vansh-Singhal-12/TerraScope-AI)

I acknowledge the original creators of all datasets, pretrained weights, libraries, and external resources used in this project.