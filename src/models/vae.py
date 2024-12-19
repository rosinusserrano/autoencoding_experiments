"""Variational Autoencoder."""

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from models.autoencoder import AutoencoderConfig
from models.base import VAEXPModel
from utils.evaluate import EvalMode
from utils.nn import ResidualBlock, downsample_conv, upsample_conv
from utils.visuals import show_side_by_side

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VAEConfig(AutoencoderConfig):
    """Configuration for Variational Autoencoder."""

    name: str = "VAE"
    kld_weight: float = 0.001
    n_augmentation: int = 1


class VAE(VAEXPModel):
    """Variational Autoencoder."""

    def __init__(self, config: VAEConfig) -> None:
        """Initilize VAE."""
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
                config.latent_channels * 2,
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

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reconstruct input while applying reparameterization trick.

        This forward function supports to generate multiple latents from a
        encoded mean and logvar which will be useful for later experiments.
        """
        encoded = self.encoder(tensor)
        mean, logvar = torch.chunk(encoded, chunks=2, dim=1)

        batch_size, latent_channels, height, width = logvar.shape

        # Create isotropic noise for reparameterization trick
        isotropic_noise = torch.randn(
            batch_size,
            self.config.n_augmentation,
            latent_channels,
            height,
            width,
        )

        # Reparameterization trick
        z = mean[:, None] + isotropic_noise.to(DEVICE) * torch.exp(
            0.5 * logvar[:, None],
        )

        # Flatten augmentation dimension to feed into decoder
        z_flattened = z.reshape(
            batch_size * self.config.n_augmentation,
            latent_channels,
            height,
            width,
        )

        decoded = self.decoder(z_flattened)

        # Reshape back to separate augmentation dimension
        decoded = decoded.reshape(
            batch_size,
            self.config.n_augmentation,
            *decoded.shape[1:],
        ).squeeze(dim=1)

        if self.config.return_latents:
            return decoded, mean, logvar, z

        return decoded, mean, logvar

    def visualize_test_batch(
        self,
        testbatch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reconstructs test batch and generates new samples."""
        with EvalMode(self):
            images = testbatch[0]
            images = images[:8].to(DEVICE)
            reconstructions = self.forward(images)[0]
            side_by_side = show_side_by_side(images, reconstructions)

            img_height, img_width = images.shape[-2:]
            downscale_factor = 2 ** (
                len(self.config.downsampling_channels) - 1
            )
            latent_height = img_height / downscale_factor
            latent_width = img_width / downscale_factor
            prior_latents = torch.randn(
                8,
                self.config.latent_channels,
                latent_height,
                latent_width,
            ).to(DEVICE)
            generations = (self.decoder(prior_latents) + 1) / 2

            return {
                "VAE_reconstructions": side_by_side,
                "VAE_generations": generations,
            }


def mse_and_kld_loss(
    model_outputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    model_config: VAEConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute sum of mse loss and weighted kld loss."""
    if model_config.return_latents:
        decoded, mean, logvar, latents = model_outputs
    else:
        decoded, mean, logvar = model_outputs

    if model_config.n_augmentation > 1:
        target = target[:, None]

    mse = F.mse_loss(decoded, target)
    kld = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=(1, 2, 3)),
        dim=0,
    )

    loss = mse + model_config.kld_weight * kld

    return loss, {
        "kld_loss_weighted": model_config.kld_weight * kld.item(),
        "kld_loss": kld.item(),
        "mse_loss": mse.item(),
        "total_loss": loss.item(),
    }
