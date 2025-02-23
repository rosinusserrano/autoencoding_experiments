"""Logger for MLFlow."""

from typing import Literal

import mlflow
import torch
from mlflow.pytorch import log_state_dict
from torchvision.utils import make_grid

from logger.base import Logger


class MlFlowLogger(Logger):
    """Logger for MLFlow."""

    def __init__(
        self,
        experiment_name: str,
        remote_url: str,
        *,
        debug: bool = False,
    ) -> None:
        """Initialize MLFlow logger."""
        self.experiment_name = experiment_name
        self.remote_url = remote_url
        self.debug = debug

        self.print("Setting tracking uri for mlflow")
        mlflow.set_tracking_uri(self.remote_url)
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.print(
            f"Created experiment with id {self.experiment.experiment_id}",
        )
        self.print("Starting run")
        mlflow.start_run()

    def print(self, *msg: str) -> None:
        """Print if in debug mode."""
        if self.debug:
            print(*msg)  # noqa: T201

    def log_configs(self, configs: dict[str, dict]) -> None:
        """Log configs, usually at beginning of training."""
        self.print("Logging configs.")
        for config_name, config_dict in configs.items():
            mlflow.log_dict(config_dict, f"{config_name}.yaml")
            mlflow.log_params(
                {f"{config_name}.{k}": v for k, v in config_dict.items()},
            )

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
    ) -> None:
        """Log metric to mlflow."""
        self.print("Logging", metric_name, metric_value)
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
        self.print("Logging images for epoch", epoch)
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

        mlflow.log_image(
            image=final_image,
            artifact_file=f"{title}_{epoch:3d}.png",
        )

    def log_message(self, message: str) -> None:
        """Log simple message into some file."""
        mlflow.log_text(message, "stdout.txt")

    def save(
        self,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Save model and optimizer to mlflow."""
        self.print("Saving model and optimizer.")
        if model is not None:
            log_state_dict(model.state_dict(), "model")
        if optimizer is not None:
            log_state_dict(optimizer.state_dict(), "optimizer")

        if optimizer is None and model is None:
            msg = "Using save function without optimizer nor model."
            raise UserWarning(msg)

    def wrapup(self) -> None:
        """Close mlflow run."""
        self.print("Closing run.")
        mlflow.end_run()
