"""Util functions for training."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    """Configuration for a autoencoder training pipeline."""

    optimizer: Literal["adam", "sgd"]
    learning_rate: float
    n_epochs: int


def create_optimizer(
    train_config: TrainConfig,
    model: nn.Module,
) -> Optimizer:
    """Create a optimizer with given config for given model."""
    if train_config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
        )

    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    reconstruct: bool = False,
) -> float:
    """Train a model for one epoch."""
    losses = []

    for batch, labels in train_loader:
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)  # noqa: PLW2901
        if reconstruct:
            labels = batch  # noqa: PLW2901

        optimizer.zero_grad()

        preds = model(batch)
        loss = loss_fn(preds, labels)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    return sum(losses) / len(losses)
