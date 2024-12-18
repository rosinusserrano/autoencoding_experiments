"""Module for getting datasets."""

from dataclasses import dataclass
from typing import Literal

from torch.utils.data import DataLoader, random_split

from datasets.cifar10 import get_cifar10_datasets
from datasets.stl10 import get_stl10_datasets


@dataclass
class DatasetConfig:
    """Configuration for dataset fetching."""

    dataset_name: Literal["cifar10", "stl10"]
    validation_split: float | None = None
    batch_size: int = 32
    subset_ratio: float | None = None


def load_data(
    config: DatasetConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Fetch datasets and put into dataloaders."""
    if config.dataset_name == "cifar10":
        datasets = get_cifar10_datasets(
            validation_split=config.validation_split,
        )
        train_set, validation_set, test_set = datasets
    elif config.dataset_name == "stl10":
        datasets = get_stl10_datasets(
            validation_split=config.validation_split,
        )
    else:
        msg = "Only cifar and stl dataset implemented until now."
        raise NotImplementedError(msg)

    if config.subset_ratio is not None:
        if config.subset_ratio > 1 or config.subset_ratio < 0:
            msg = "subset_ratio should be between 0 and 1."
            raise ValueError(msg)

        split = [config.subset_ratio, 1 - config.subset_ratio]
        train_set, _ = random_split(train_set, split)
        if validation_set is not None:
            validation_set, _ = random_split(validation_set, split)
        test_set, _ = random_split(test_set, split)

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
