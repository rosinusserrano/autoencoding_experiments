"""Function to fetch STL10 dataset."""

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import STL10
from torchvision.transforms import v2


def get_stl10_datasets(
    root: str = "data",
    validation_split: float | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Get STL10 dataset."""
    train_set = STL10(
        root,
        split="unlabeled",
        transform=v2.Compose(
            [
                v2.ToTensor(),
                v2.RandomResizedCrop((96, 96), scale=(0.5, 1)),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        ),
        download=True,
    )

    validation_set = None
    if validation_split is not None:
        if validation_split <= 0 or validation_split >= 1:
            msg = "Validation split should be in interval ]0, 1["
            raise ValueError(msg)
        train_set, validation_set = random_split(
            train_set,
            [1 - validation_split, validation_split],
        )

    test_set = STL10(
        root,
        split="test",
        transform=v2.ToTensor(),
        download=True,
    )

    return train_set, validation_set, test_set
