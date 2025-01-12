"""Autoencoder implementation."""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import nn

from models.base import ModelConfig, VAEXPModel
from utils.evaluate import EvalMode
from utils.nn import (
    ResidualBlock,
    downsample_conv,
    lin_act_norm,
    upsample_conv,
)
from utils.visuals import show_side_by_side

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AutoencoderConfig(ModelConfig):
    """Config for Autoencoder."""

    name: str = field(default="Autoencoder", metadata={"fixed": True})

    # Encoder
    downsampling_channels: list[int] = field(
        default_factory=lambda: [3, 256, 256],
        metadata={"help": "Downsampling channels"},
    )
    encoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256],
        metadata={"help": "Encoder residual channels"},
    )

    # Bottleneck
    latent_channels: int = field(
        default=256,
        metadata={"help": "Latent channels"},
    )

    # Decoder
    decoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256],
        metadata={"help": "Decoder residual channels"},
    )
    upsampling_channels: list[int] = field(
        default_factory=lambda: [256, 256, 3],
        metadata={"help": "Upsampling channels"},
    )

    # Execution behaviour
    return_latents: bool = field(default=False, metadata={"help": ""})


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

    def visualize_test_batch(
        self,
        testbatch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reconstructs test batch and generates new samples."""
        with EvalMode(self):
            images = testbatch[0].to(DEVICE)
            images = images[:8]
            reconstructions = self.forward(images)
            side_by_side = show_side_by_side(images, reconstructions)

            return {
                "AE_Reconstructions": side_by_side,
            }


def mse_loss(inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mse loss loss return with dict for stats."""
    loss = nn.functional.mse_loss(inp, target)
    return loss, {"mse": loss.item()}


@dataclass
class AutoencoderV2Config(ModelConfig):
    """Config for the AE v2."""

    name: str = field(default="AutoencoderV2", metadata={"fixed": True})
    description: str = field(
        default="Upgrade to the fully convolutional autoencoder "
        "where there is a fully connected bottleneck to produce a more "
        "compact but meaningful latent representation.",
        metadata={"fixed": True},
    )

    input_shape: list[int] = field(
        default_factory=lambda: [3, 96, 96],
        metadata={
            "help": "Expected input shape for this model."
            " Important because the fully connected bottleneck causes the"
            " model not to wor for varying input shapes.",
        },
    )

    # Encoder
    downsampling_channels: list[int] = field(
        default_factory=lambda: [3, 256, 256],
        metadata={
            "help": "Specifies number of downsampling layers. First value is"
            " number of input channels, last value is number of output"
            " channels of downsampling part.",
        },
    )
    encoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 16],
        metadata={
            "help": "Encoder residual channels.",
        },
    )
    encoder_fc_features: list[int] = field(
        default_factory=lambda: [9216, 4096, 1024],
        metadata={
            "help": "Encoder fully connected features.",
        },
    )

    # Bottleneck
    latent_channels: int = field(
        default=256, metadata={"help": "Latent channels"}
    )

    # Decoder
    decoder_fc_features: list[int] = field(
        default_factory=lambda: [1024, 4096, 9216],
        metadata={
            "help": "Decoder fully connected features",
        },
    )
    decoder_residual_channels: list[int] = field(
        default_factory=lambda: [16, 256, 256],
        metadata={
            "help": "Decoder residual channels",
        },
    )
    upsampling_channels: list[int] = field(
        default_factory=lambda: [256, 256, 3],
        metadata={
            "help": "Upsampling channels",
        },
    )


class AutoencoderV2(VAEXPModel):
    """A basic implementation of an autoencoder."""

    def __init__(self, config: AutoencoderV2Config) -> None:
        """Instantiate the autoencoder class."""
        super().__init__()

        self.config = config

        self.encoder_conv = nn.Sequential(
            *[
                downsample_conv(inc, outc)
                for inc, outc in pairwise(config.downsampling_channels)
            ],
            *[
                ResidualBlock(inc, outc)
                for inc, outc in pairwise(config.encoder_residual_channels)
            ],
        )

        self.encoder_fc = nn.Sequential(
            *[
                lin_act_norm(in_feats, out_feats)
                for in_feats, out_feats in pairwise(config.encoder_fc_features)
            ],
        )

        self.decoder_fc = nn.Sequential(
            *[
                lin_act_norm(in_feats, out_feats)
                for in_feats, out_feats in pairwise(config.decoder_fc_features)
            ],
        )

        self.decoder_conv = nn.Sequential(
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
        """Reconstruct images with the Autoencoder.

        Assumes encoder fc and decoder fc are mirrored.
        """
        out = self.encoder_conv(tensor)
        bs, c, h, w = out.shape
        out = torch.flatten(out, start_dim=1)
        out = self.encoder_fc(out)
        out = self.decoder_fc(out)
        out = torch.reshape(out, (bs, c, h, w))
        out = self.decoder_conv(out)

        return out

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct images as in forward, but in eval mode."""
        with EvalMode(self):
            return self.forward(images)

    def visualize_test_batch(
        self,
        testbatch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reconstructs test batch and generates new samples."""
        with EvalMode(self):
            images = testbatch[0].to(DEVICE)
            images = images[:8]
            reconstructions = self.forward(images)
            side_by_side = show_side_by_side(images, reconstructions)

            return {
                "AE_Reconstructions": side_by_side,
            }
