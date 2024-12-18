"""Util functions for training."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from datasets import DatasetConfig, load_data
from logger.base import Logger
from models import ModelConfig, create_model
from utils.evaluate import evaluate
from utils.nn import LossFn
from utils.stats import append_dict_values, average_dict_values

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
    loss_fn: LossFn,
    *,
    reconstruct: bool = False,
) -> dict[str, float]:
    """Train a model for one epoch."""
    stats = {}

    for batch, labels in train_loader:
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)  # noqa: PLW2901
        if reconstruct:
            labels = batch  # noqa: PLW2901

        optimizer.zero_grad()

        preds = model(batch)
        loss, batch_stats = loss_fn(preds, labels)
        loss.backward()

        append_dict_values(stats, batch_stats)

        optimizer.step()

    return average_dict_values(stats)


def standard_training_pipeline(  # noqa: PLR0913
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    logger: Logger,
    loss_fn: LossFn,
    validation_interval: int | None = 1,
    test_interval: int | None = None,
    visualization_interval: int | None = None,
) -> None:
    """Train a simple model."""
    model = create_model(model_config)
    train_loader, val_loader, test_loader = load_data(dataset_config)
    optimizer = create_optimizer(train_config=train_config, model=model)

    logger.log_configs(
        {
            "model": model_config,
            "training": train_config,
            "dataset": dataset_config,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.log_message(f"On device: {device}")

    for epoch in range(train_config.n_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            reconstruct=True,
        )
        for key, value in train_stats.items():
            logger.log_metric(f"{key}_train", value, epoch)

        if epoch % validation_interval == 0:
            val_stats = evaluate(
                model,
                val_loader,
                loss_fn,
                reconstruct=True,
            )
            for key, value in val_stats.items():
                logger.log_metric(f"{key}_val", value, epoch)

        if test_interval is not None and epoch % test_interval == 0:
            test_stats = evaluate(
                model,
                test_loader,
                loss_fn,
                reconstruct=True,
            )
            for key, value in test_stats.items():
                logger.log_metric(f"{key}_test", value, epoch)

        if (
            visualization_interval is not None
            and epoch % visualization_interval == 0
        ):
            testbatch = next(iter(val_loader))
            visualizations = model.visualize_test_batch(testbatch)
            for title, imgtensor in visualizations.items():
                logger.log_image_tensor(
                    images=imgtensor,
                    title=title,
                )
