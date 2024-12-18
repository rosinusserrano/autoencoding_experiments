"""Basically just the get_model() function and imports."""

from models.autoencoder import Autoencoder, AutoencoderConfig
from models.base import ModelConfig, VAEXPModel
from models.vae import VAE, VAEConfig


def create_model(config: ModelConfig) -> VAEXPModel:
    """Return the correct model for a given config."""
    if config.name == "Autoencoder":
        return Autoencoder(config)
    if config.name == "VAE":
        return VAE(config)

    msg = f"Config ({config}) is not supported by the `get_model()` function."
    raise ValueError(msg)
