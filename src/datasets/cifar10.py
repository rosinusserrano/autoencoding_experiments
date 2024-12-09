"""Module for Cifar 10 dataset fetching."""

from collections.abc import Callable

from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10


def get_cifar10_datasets(
    root: str = "data",
    validation_split: float | None = None,
    train_transforms: Callable | None = None,
    test_transforms: Callable | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Get CIFAR 10 dataset as PyTorch `Dataset`.

    Params:
    - `validation_split`: If provided, returns an additional validation set
    """
    train_set = CIFAR10(
        root,
        train=True,
        transform=train_transforms,
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

    test_set = CIFAR10(
        root,
        train=False,
        transform=test_transforms,
        download=True,
    )

    if validation_split is not None:
        return train_set, validation_set, test_set

    return train_set, test_set
