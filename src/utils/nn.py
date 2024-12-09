"""Utility functions and classes for neural network design."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class PrintModule(nn.Module):
    """Module for debugging.

    Prints the shape of its input and simply forwards it.
    """

    def __init__(self) -> None:
        """Return PrintModule for debugging."""
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Print shape of input and return it."""
        print(tensor.shape)  # noqa: T201
        return tensor


class ResidualBlock(nn.Module):
    """Create a residual convolutional block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a residual convolutional block."""
        super().__init__()

        self.conv_block = nn.Sequential(
            conv_act_norm(in_channels, out_channels),
            conv_act_norm(out_channels, out_channels),
            conv_act_norm(out_channels, out_channels),
        )

        self.use_skip_conv = in_channels != out_channels
        self.skip_conv = (
            None
            if self.use_skip_conv
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add input to output of conv block, use skip conv if necessary."""
        out = self.conv_block(tensor)

        if self.use_skip_conv:
            out = self.skip_conv(tensor) + out
        else:
            out = tensor + out

        return out


def downsample_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create a convolutional layer appropriate for downsampling."""
    return conv_act_norm(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )


def upsample_conv(
    in_channels: int,
    out_channels: int,
    *,
    activation: str = "relu",
    use_batchnorm: bool = True,
) -> nn.Sequential:
    """Create a deconvolutional block appropriate for upsampling."""
    layers = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        get_activation_by_name(activation),
    ]

    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv_act_norm(  # noqa: PLR0913
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    activation: str = "relu",
) -> nn.Sequential:
    """Create the basic convolution -> activation -> normalization module."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        get_activation_by_name(activation),
        nn.BatchNorm2d(out_channels),
    )


def get_activation_by_name(name: str) -> nn.Module | Callable:
    """Return the activation module given a name."""
    if name == "relu":
        return nn.ReLU()

    if name == "tanh":
        return nn.Tanh()

    msg = f"The activation '{name}' isn't defined"
    raise ValueError(msg)


LossFnName = Literal[
    "mse",
    "cross_entropy",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
]


def get_loss_fn_by_name(
    name: LossFnName,
) -> Callable:
    """Return a function specified by the name string."""
    if name == "mse":
        return F.mse_loss
    if name == "cross_entropy":
        return F.cross_entropy
    if name == "binary_cross_entropy":
        return F.binary_cross_entropy
    if name == "binary_cross_entropy_with_logits":
        return F.binary_cross_entropy_with_logits

    msg = f"Activation with name {name} isn't supported yet sorry :)"
    raise NotImplementedError(msg)
