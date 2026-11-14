"""ResNet-18 architectures used by TerraScope AI."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, initialization: str = "scratch"):
    """Build the project's ResNet-18 variant for 32x32 satellite images.

    Args:
        num_classes: Number of output classes.
        initialization: "scratch" or "imagenet".

    The 7x7 stem convolution is retained, but its stride is changed to 1
    and the max-pooling layer is removed to preserve spatial information
    in 32x32 inputs.
    """
    if initialization == "imagenet":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif initialization == "scratch":
        model = resnet18(weights=None)
    else:
        raise ValueError(
            f"initialization must be 'scratch' or 'imagenet', got {initialization!r}"
        )

    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_eurosat_backbone(model, checkpoint_path: str):
    """Load EuroSAT-trained feature-extractor weights into a target model.

    The EuroSAT classifier head is intentionally excluded because So2Sat
    uses a different label space (17 LCZ classes rather than 10 EuroSAT
    land-cover classes).
    """
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model_state_dict"]
    backbone_state = {
        key: value for key, value in state.items()
        if not key.startswith("fc.")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)

    # Only the new classifier head should be missing.
    if any(not key.startswith("fc.") for key in missing) or unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch. Missing={missing}, "
            f"unexpected={unexpected}"
        )
    return checkpoint
