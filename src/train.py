"""Functions and classes to train models."""

from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
        if reconstruct:
            labels = batch  # noqa: PLW2901

        optimizer.zero_grad()

        preds = model(batch)
        loss = loss_fn(preds, labels)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

    return sum(losses) / len(losses)
