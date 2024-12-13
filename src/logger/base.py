"""Abstract base class for logging modules."""

from abc import ABC, abstractmethod
from typing import Literal

import torch


class Logger(ABC):
    """Abstact base class for loggers."""

    @abstractmethod
    def log_configs(self, configs: dict[str, dict]) -> None:
        """Log configs (probably done initally).

        Gets a dict of dicts where th keys are the names of the config.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
    ) -> None:
        """Log a metric for a given epoch, which has a name and a value."""
        raise NotImplementedError

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

    def log_grouped_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int,
        group: str,
    ) -> None:
        """Print metric to stdout and save to metrics dict for final plot."""
        raise NotImplementedError

    @abstractmethod
    def log_image_tensor(
        self,
        images: torch.Tensor,
        title: str = "",
        method: Literal["append", "update"] = "append",
    ) -> None:
        """Log images."""
        raise NotImplementedError

    @abstractmethod
    def wrapup(self) -> None:
        """Do final things with metric and stuff."""
        raise NotImplementedError
