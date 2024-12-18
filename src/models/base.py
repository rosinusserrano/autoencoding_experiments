"""Abstract base classes for models and configs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


class VAEXPModel(nn.Module, ABC):
    """Abstract base class for all models in this project."""

    @abstractmethod
    def visualize_test_batch(self, testbatch) -> dict[str, torch.Tensor]:
        """Return a dict with image tensors and their description."""
        raise NotImplementedError


@dataclass
class ModelConfig(ABC):
    """Standard config for all models."""

    name: str
