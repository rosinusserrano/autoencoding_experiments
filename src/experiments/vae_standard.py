"""Training standard VAE."""

from functools import partial

import torch

from models.vae import VAE, VAEConfig, mse_and_kld_loss
from datasets import DatasetConfig, load_data
from utils.train import TrainConfig, train_one_epoch, create_optimizer
from utils.evaluate import evaluate, EvalMode
from utils.visuals import show_side_by_side
from logger.base import Logger


def train_standard_vae_on_cifar(
    model_config: VAEConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    logger: Logger,
    validation_interval: int = 1,
    test_interval: int | None = None,
    visualization_interval: int | None = None,
):
    """Train standard VAE."""
    if model_config.n_augmentation != 1:
        msg = "The standard VAE only supports n_augmentation = 1."

    model = VAE(model_config)
    train_loader, val_loader, test_loader = load_data(dataset_config)
    optimizer = create_optimizer(train_config, model)

    logger.log_configs(
        {
            "model": model_config,
            "dataset": dataset_config,
            "train": train_config,
        }
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.log_message(f"On device: {device}")

    for epoch in range(train_config.n_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=partial(mse_and_kld_loss, model_config=model_config),
            reconstruct=True,
        )

        logger.log_metric("train_loss", train_loss, epoch)

        if epoch % validation_interval == 0:
            val_loss = evaluate(
                model,
                val_loader,
                partial(mse_and_kld_loss, model_config=model_config),
                reconstruct=True,
            )
            logger.log_metric("val_loss", val_loss, epoch)

        if test_interval is not None and epoch % test_interval == 0:
            test_loss = evaluate(
                model,
                test_loader,
                partial(mse_and_kld_loss, model_config=model_config),
                reconstruct=True,
            )
            logger.log_metric("test_loss", test_loss, epoch)

        if (
            visualization_interval is not None
            and epoch % visualization_interval == 0
        ):
            with EvalMode(model):
                images, _ = next(iter(test_loader))
                images = images[:8].to(device)
                reconstructions = model(images)[0]
                side_by_side = show_side_by_side(images, reconstructions)
                logger.log_image_tensor(
                    images=side_by_side,
                    title="VAE_reconstructions",
                )

                prior_latents = torch.randn(
                    8, model_config.latent_channels, 8, 8
                ).to(device)
                generations = (model.decoder(prior_latents) + 1) / 2
                logger.log_image_tensor(generations, title="VAE_generations")

    logger.wrapup()
    logger.save(model, optimizer)
