"""Module for getting datasets."""

from typing import Literal

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomResizedCrop, ToTensor

from datasets.cifar10 import get_cifar10_datasets


def load_data(
    dataset_name: Literal["cifar10", "mnist", "fashionmnist"],
    validation_split: float | None = None,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Fetch datasets and put into dataloaders."""
    if dataset_name == "cifar10":
        datasets = get_cifar10_datasets(
            validation_split=validation_split,
            train_transforms=Compose(
                [ToTensor(), RandomResizedCrop((32, 32), scale=(0.2, 1))],
            ),
            test_transforms=ToTensor(),
        )
        train_set, validation_set, test_set = datasets

    else:
        msg = "Only CIFAR10 dataset implemented until now."
        raise NotImplementedError(msg)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = (
        None
        if validation_set is None
        else DataLoader(validation_set, batch_size=batch_size)
    )
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, validation_loader, test_loader
