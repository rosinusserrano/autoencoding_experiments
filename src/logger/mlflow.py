"""Logger for MLFlow."""

from typing import Literal

import mlflow
from mlflow.pytorch import save_model, save_state_dict
import torch
from torchvision.utils import make_grid

from logger.base import Logger


class MlFlowLogger(Logger):
    """Logger for MLFlow."""

    def __init__(self, experiment_name: str, remote_url: str) -> None:
        """Initialize MLFlow logger."""
        self.experiment_name = experiment_name
        self.remote_url = remote_url

        mlflow.set_tracking_uri(self.remote_url)
        self.experiment = mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()

    def log_configs(self, configs: dict[str, dict]) -> None:
        """Log configs, usually at beginning of training."""
        for config_name, config_dict in configs.items():
            mlflow.log_dict(config_dict, f"{config_name}.yaml")

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
    ) -> None:
        """Log metric to mlflow."""
        mlflow.log_metric(metric_name, metric_value, epoch)

    def log_grouped_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
        group: str,  # noqa: ARG002
    ) -> None:
        """Log grouped metric with mlflow.

        Attention: mlflow doesnt really support this, so
        """
        return self.log_metric(metric_name, metric_value, epoch)

    def log_image_tensor(
        self,
        images: torch.Tensor,
        epoch: int,
        title: str = "",
        method: Literal["append", "update"] = "append",
    ) -> None:
        """Log images to mflow."""
        if images.dim() == 4:  # batched images  # noqa: PLR2004
            final_image = (
                make_grid(
                    images,
                    padding=1,
                    normalize=True,
                )
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
        elif images.dim() == 3:  # single image  # noqa: PLR2004
            final_image = images.permute(1, 2, 0).detach().cpu().numpy()
        else:
            msg = f"Got image tensor with {images.dim()} dimensions ..?!\n"
            raise ValueError(msg)

        if method == "update":
            msg = "Method 'update' doesn't affect image uploads in mlflow."
            raise UserWarning(msg)

        mlflow.log_image(image=final_image, key=title, step=epoch)

    def log_message(self, message: str) -> None:
        """Log simple message into some file."""
        mlflow.log_text(message, "stdout.txt")

    def save(
        self,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Save model and optimizer to mlflow."""
        if model is not None:
            save_model(model, "model")
        if optimizer is not None:
            save_state_dict()

        if optimizer is None and model is None:
            msg = "Using save function without optimizer nor model."
            raise UserWarning(msg)

    def wrapup(self) -> None:
        """Close mlflow run."""
        mlflow.end_run()
