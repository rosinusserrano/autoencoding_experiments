"""Standard autoencoder."""

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F  # noqa: N812

from datasets import DatasetConfig, load_data
from logger.base import Logger
from logger.caveman import CavemanLogger
from models.autoencoder import Autoencoder, AutoencoderConfig
from utils.evaluate import EvalMode, evaluate
from utils.train import TrainConfig, create_optimizer, train_one_epoch
from utils.visuals import show_side_by_side


def train_standard_autoencoder(  # noqa: PLR0913
    model_config: AutoencoderConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    logger: Logger,
    loss_fn: F.mse_loss,
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
            loss_fn,
            reconstruct=True,
        )
        logger.log_metric("train_loss", train_loss, epoch)

        if epoch % validation_interval == 0:
            val_loss = evaluate(
                model,
                val_loader,
                loss_fn,
                reconstruct=True,
            )
            logger.log_metric("val_loss", val_loss, epoch)

        if test_interval is not None and epoch % test_interval == 0:
            test_loss = evaluate(
                model,
                test_loader,
                loss_fn,
                reconstruct=True,
            )
            logger.log_metric("test_loss", test_loss, epoch)

        if (
            visualization_interval is not None
            and epoch % visualization_interval == 0
        ):
            images, _ = next(iter(val_loader))
            images = images[:8].to(device)
            reconstructions = model.reconstruct(images)

            side_by_side = show_side_by_side(images, reconstructions)

            logger.log_image_tensor(
                images=side_by_side,
                title="AE_reconstructions",
            )

    logger.wrapup()
    logger.save(model, optimizer)

    return model


def gridsearch_bs_and_lr(logging_dir: str) -> None:
    """Run experiment: standard autoencoder reconstruction."""
    for bs in [32, 64, 128]:
        for lr in [0.1, 0.001, 0.0001]:
            model_config = AutoencoderConfig()

            dataset_config = DatasetConfig(
                dataset_name="cifar10",
                batch_size=bs,
                validation_split=0.1,
            )

            train_config = TrainConfig(
                optimizer="adam",
                learning_rate=lr,
                n_epochs=10,
            )

            cmlogger = CavemanLogger(
                root_dir=logging_dir,
                experiment_name=f"AE_cifar_bs-{bs}_lr-{lr}",
            )

            train_standard_autoencoder(
                model_config=model_config,
                dataset_config=dataset_config,
                train_config=train_config,
                logger=cmlogger,
                validation_interval=1,
                test_interval=1,
                visualization_interval=1,
            )


def latent_walk_animation(
    model_fp: Path,
    model_config_fp: Path,
    n_images: int,
    n_walk: int,
    destination: Path,
    fps: int = 25,
):
    """Animate inteprolation between latents."""
    dataset_config = DatasetConfig(
        dataset_name="cifar10",
        validation_split=None,
        batch_size=n_images,
    )

    loader = load_data(dataset_config)[2]
    images, _ = next(iter(loader))

    with Path.open(model_config_fp, encoding="utf-8") as model_config_file:
        model_config_dict = yaml.safe_load(model_config_file)
        model_config = AutoencoderConfig(**model_config_dict)
        model = Autoencoder(model_config)
        model.load_state_dict(
            torch.load(
                model_fp,
                weights_only=True,
                map_location="cpu",
            ),
        )

    with EvalMode(model):
        latents = model.encoder(images)

    generated_images = torch.zeros(
        ((n_images - 1) * n_walk, 3, 32, 32),
    )

    for i in range(n_images - 1):
        interpolation_coeffs = torch.linspace(0, 1, n_walk)
        interpolation_coeffs = interpolation_coeffs[:, None, None, None]

        generated_images[i * n_walk : (i + 1) * n_walk] = model.decoder(
            latents[i] * (1 - interpolation_coeffs)
            + latents[i + 1] * interpolation_coeffs,
        )

    generated_images = generated_images / 2 + 0.5
    generated_images = generated_images * 255
    generated_images = torch.clamp(generated_images, 0, 255).long()

    frames = [
        Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8))
        for img in generated_images.detach()
    ]

    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
