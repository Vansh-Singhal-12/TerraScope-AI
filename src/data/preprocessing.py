"""Image preprocessing utilities for TerraScope AI."""

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SENTINEL2_RGB_BANDS = ("B4", "B3", "B2")


def preprocess_eurosat(image):
    """EuroSAT PIL images are resized and normalized in the dataset transform."""
    raise NotImplementedError(
        "Use the EuroSAT transform defined in src/data/eurosat.py."
    )


def preprocess_sentinel2_rgb(
    image: np.ndarray,
    reflectance_max: float = 2.8,
) -> torch.Tensor:
    """Convert Sentinel-2 RGB reflectance (H,W,3) to normalized (3,H,W).

    The input channel order must be B4, B3, B2 (red, green, blue).
    Values are clipped to the documented project range [0, 2.8], scaled
    to [0, 1], then normalized with ImageNet mean and standard deviation.

    This is the project's chosen consistent preprocessing convention,
    not a claim that ImageNet normalization is optimal for reflectance data.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected an HxWx3 RGB array, received shape {image.shape}"
        )
    if reflectance_max <= 0:
        raise ValueError("reflectance_max must be positive")

    image = np.clip(image, 0.0, reflectance_max) / reflectance_max
    tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std
