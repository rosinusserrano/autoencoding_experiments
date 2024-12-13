"""Autoencoder implementation."""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from datasets import DatasetConfig, load_data
from logger.base import Logger
from utils.evaluate import EvalMode, evaluate
from utils.nn import ResidualBlock, downsample_conv, upsample_conv
from utils.train import TrainConfig, create_optimizer, train_one_epoch
from utils.visuals import show_side_by_side


@dataclass
class AutoencoderConfig:
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
    latent_channels: int = 16

    # Decoder
    decoder_residual_channels: list[int] = field(
        default_factory=lambda: [256, 256, 256],
    )
    upsampling_channels: list[int] = field(
        default_factory=lambda: [256, 256, 3],
    )


class Autoencoder(nn.Module):
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
        return self.decoder(self.encoder(tensor))

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct images as in forward, but in eval mode."""
        with EvalMode(self):
            return self.forward(images)


def run(  # noqa: PLR0913
    model_config: AutoencoderConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    logger: Logger,
    validation_interval: int = 1,
    test_interval: int | None = None,
    visualization_interval: int | None = None,
) -> Autoencoder:
    """Train the autoencoder."""
    model = Autoencoder(model_config)
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
    print(f"On device: {device}")

    for epoch in range(train_config.n_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            F.mse_loss,
            reconstruct=True,
        )
        logger.log_metric("train_loss", train_loss, epoch)

        if epoch % validation_interval == 0:
            val_loss = evaluate(
                model,
                val_loader,
                F.mse_loss,
                reconstruct=True,
            )
            logger.log_metric("val_loss", val_loss, epoch)

        if test_interval is not None and epoch % test_interval == 0:
            test_loss = evaluate(
                model,
                test_loader,
                F.mse_loss,
                reconstruct=True,
            )
            logger.log_metric("test_loss", test_loss, epoch)

        if (
            visualization_interval is not None
            and epoch % visualization_interval == 0
        ):
            images, _ = next(iter(val_loader))
            images = images[:8]
            reconstructions = model.reconstruct(images)

            side_by_side = show_side_by_side(images, reconstructions)

            logger.log_image_tensor(
                images=side_by_side,
                title="AE_reconstructions",
            )

    return model
