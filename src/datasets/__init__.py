"""Module for getting datasets."""

from dataclasses import dataclass
from typing import Literal

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    RandomResizedCrop,
    ToTensor,
    Normalize,
)

from datasets.cifar10 import get_cifar10_datasets


@dataclass
class DatasetConfig:
    """Configuration for dataset fetching."""

    dataset_name: Literal["cifar10", "mnist", "fashionmnist"]
    validation_split: float | None = None
    batch_size: int = 32


def load_data(
    config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Fetch datasets and put into dataloaders."""
    if config.dataset_name == "cifar10":
        datasets = get_cifar10_datasets(
            validation_split=config.validation_split,
            train_transforms=Compose(
                [
                    ToTensor(),
                    RandomResizedCrop((32, 32), scale=(0.2, 1)),
                    Normalize(mean=[125, 125, 125], std=[125, 125, 125]),
                ],
            ),
            test_transforms=Compose(
                [
                    ToTensor(),
                    Normalize(mean=[125, 125, 125], std=[125, 125, 125]),
                ],
            ),
        )
        train_set, validation_set, test_set = datasets

    else:
        msg = "Only CIFAR10 dataset implemented until now."
        raise NotImplementedError(msg)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = (
        None
        if validation_set is None
        else DataLoader(validation_set, batch_size=config.batch_size)
    )
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    return train_loader, validation_loader, test_loader
