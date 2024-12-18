"""Functions and classes to evaluate models."""

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.stats import append_dict_values, average_dict_values

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EvalMode:
    """Put a model in eval mode and activate no grad context."""

    def __init__(self, model: nn.Module) -> None:
        """Put a model in eval mode and activate no grad context."""
        self.grad_previously_enabled = False
        self.previous_training_mode = model.training
        self.model = model

    def __enter__(self) -> None:
        """Set model in eval mode and activate no grad env."""
        self.grad_previously_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self.model.train(mode=False)

    def __exit__(self, *args: object) -> None:
        """Set model back to previous train mode and deactive no grad mode."""
        torch.set_grad_enabled(self.grad_previously_enabled)
        self.model.train(self.previous_training_mode)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    reconstruct: bool = False,
) -> float:
    """Train a model for one epoch."""
    stats = {}

    with EvalMode(model):
        for batch, labels in test_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)  # noqa: PLW2901
            if reconstruct:
                labels = batch  # noqa: PLW2901

            preds = model(batch)
            loss, batch_stats = loss_fn(preds, labels)
            append_dict_values(stats, batch_stats)

    return average_dict_values(stats)
