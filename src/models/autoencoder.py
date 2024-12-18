"""Autoencoder implementation."""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import nn

from models.base import ModelConfig, VAEXPModel
from utils.evaluate import EvalMode
from utils.nn import ResidualBlock, downsample_conv, upsample_conv


@dataclass
class AutoencoderConfig(ModelConfig):
    """Config for Autoencoder."""

    name: str = "Autoencoder"

    # Encoder
    downsampling_channels: list[int] = field(
        default_factory=lambda: [3, 256, 256],
    )
    encoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256],
    )

    # Bottleneck
    latent_channels: int = 256

    # Decoder
    decoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256],
    )
    upsampling_channels: list[int] = field(
        default_factory=lambda: [256, 256, 3],
    )

    # Execution behaviour
    return_latents: bool = False



class Autoencoder(VAEXPModel):
    """A basic implementation of an autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        """Instantiate the autoencoder class."""
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            *[
                downsample_conv(inc, outc)
                for inc, outc in pairwise(config.downsampling_channels)
            ],
            *[
                ResidualBlock(inc, outc)
                for inc, outc in pairwise(config.encoder_residual_channels)
            ],
            ResidualBlock(
                config.encoder_residual_channels[-1],
                config.latent_channels,
            ),
        )

        self.decoder = nn.Sequential(
            ResidualBlock(
                config.latent_channels,
                config.decoder_residual_channels[0],
            ),
            *[
                ResidualBlock(inc, outc)
                for inc, outc in pairwise(config.decoder_residual_channels)
            ],
            *[
                upsample_conv(inc, outc)
                for inc, outc in pairwise(config.upsampling_channels[:-1])
            ],
            upsample_conv(
                config.upsampling_channels[-2],
                config.upsampling_channels[-1],
                activation="tanh",
                use_batchnorm=False,
            ),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct images with the Autoencoder."""
        latents = self.encoder(tensor)

        if self.config.return_latents:
            return self.decoder(latents), latents

        return self.decoder(latents)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct images as in forward, but in eval mode."""
        with EvalMode(self):
            return self.forward(images)
