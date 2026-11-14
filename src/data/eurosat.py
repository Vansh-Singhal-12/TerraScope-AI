"""EuroSAT dataset and reproducible train/validation split."""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT


def create_eurosat_loaders(
    root: str,
    batch_size: int = 64,
    random_state: int = 42,
    num_workers: int = 0,
):
    """Create the stratified 80/20 EuroSAT train/validation loaders."""
    from sklearn.model_selection import train_test_split

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = EuroSAT(root=root, download=True, transform=transform)
    labels = np.asarray(dataset.targets)

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_indices, val_indices
