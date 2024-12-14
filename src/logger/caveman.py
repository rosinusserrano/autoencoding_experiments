"""Most basic logger: print to stdout and save to filesystem."""

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
import yaml
from human_id import generate_id
from torch import nn
from torch.optim import Optimizer
from torchvision.utils import make_grid

from logger.base import Logger


class CavemanLogger(Logger):
    """Most basic logger: print to stdout and save to filesystem."""

    def __init__(self, root_dir: str | Path) -> None:
        """Most basic logger: print to stdout and save to filesystem."""
        self.root_dir = Path(root_dir)
        if not Path.exists(self.root_dir):
            Path.mkdir(self.root_dir, parents=True)

        run_index = len(os.listdir(self.root_dir))
        human_id = generate_id(word_count=3)
        run_id = f"{run_index:03d}-{human_id}"
        self.run_dir = Path(f"{root_dir}/{run_id}")

        self.plots_dir = Path(f"{self.run_dir}/plots")
        if not Path.exists(self.plots_dir):
            Path.mkdir(self.plots_dir, parents=True)

        self.image_dir = Path(f"{self.run_dir}/images")
        if not Path.exists(self.image_dir):
            Path.mkdir(self.image_dir, parents=True)

        self.config_dir = Path(f"{self.run_dir}/configs")
        if not Path.exists(self.config_dir):
            Path.mkdir(self.config_dir, parents=True)

        self.metrics: dict[str, list[float]] = {}
        self.grouped_metrics: dict[str, dict[str, list[float]]] = {}

    def log_configs(self, configs: dict[str, Any]) -> None:
        """Write configs into yaml files."""
        for name, config in configs.items():
            if isinstance(config, dict):
                with Path.open(
                    Path(f"{self.config_dir}/{name}.yaml"),
                    "w",
                    encoding="utf-8",
                ) as yaml_file:
                    yaml.dump(config, yaml_file)

            elif is_dataclass(config):
                with Path.open(
                    Path(f"{self.config_dir}/{name}.yaml"),
                    "w",
                    encoding="utf-8",
                ) as yaml_file:
                    yaml.dump(asdict(config), yaml_file)

            else:
                msg = "Config should be either dict or dataclass"
                raise NotImplementedError(msg)

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
    ) -> None:
        """Print metric to stdout and save to metrics dict for final plot."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = [metric_value]
            self.metrics[f"{metric_name}_epochs"] = [epoch]
        else:
            self.metrics[metric_name].append(metric_value)
            self.metrics[f"{metric_name}_epochs"].append(epoch)

        print(f"EPOCH {epoch}: {metric_name} {metric_value:.4f}")  # noqa: T201

    def log_grouped_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
        group: str,
    ) -> None:
        """Print metric to stdout and save to metrics dict for final plot."""
        if group not in self.grouped_metrics:
            self.grouped_metrics[group] = {}

        if metric_name not in self.grouped_metrics[group]:
            self.grouped_metrics[group][metric_name] = [metric_value]
            self.grouped_metrics[group][f"{metric_name}_epochs"] = [epoch]
        else:
            self.grouped_metrics[group][metric_name].append(metric_value)
            self.grouped_metrics[group][f"{metric_name}_epochs"].append(epoch)

    def log_image_tensor(
        self,
        images: torch.Tensor,
        title: str = "",
        method: Literal["update", "append"] = "append",
    ) -> None:
        """Save images as grid to PNG file in img directory."""
        if images.dim() == 4:  # batched images  # noqa: PLR2004
            final_image = make_grid(
                images,
                padding=1,
                normalize=True,
            ).permute(1, 2, 0)
        elif images.dim() == 3:  # single image  # noqa: PLR2004
            final_image = images.permute(1, 2, 0)
        else:
            msg = f"Got image tensor with {images.dim()} dimensions ..?!\n"
            raise ValueError(msg)

        filename = title

        if method == "append":
            index = len(
                list(filter(lambda f: title in f, os.listdir(self.image_dir))),
            )
            filename = f"{filename}_{index}"

        plt.imsave(
            f"{self.image_dir}/{filename}.png",
            final_image.detach().cpu().numpy(),
        )

    def wrapup(self) -> None:
        """Plot metrics with matplotlib into image dir."""
        for metric_name in self.metrics:
            if metric_name.endswith("_epochs"):
                continue
            values = self.metrics[metric_name]
            epochs = self.metrics[f"{metric_name}_epochs"]

            plt.clf()
            plt.plot(epochs, values)
            plt.title(metric_name)
            plt.savefig(f"{self.plots_dir}/{metric_name}")
        
        if len(self.grouped_metrics.items()) > 0:
            print("Grouped metrics not yet supported sorry.")

    def save(
        self,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        """Save model and optimizer,both optional."""
        if model is not None:
            torch.save(model.state_dict(), f"{self.run_dir}/model.pth")

        if optimizer is not None:
            torch.save(optimizer.state_dict(), f"{self.run_dir}/optimizer.pth")
