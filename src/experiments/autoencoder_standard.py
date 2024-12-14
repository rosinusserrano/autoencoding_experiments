"""Standard autoencoder."""

import torch
from torch.nn import functional as F  # noqa: N812

from datasets import DatasetConfig, load_data
from logger.base import Logger
from logger.caveman import CavemanLogger
from models.autoencoder import Autoencoder, AutoencoderConfig
from utils.evaluate import evaluate
from utils.train import TrainConfig, create_optimizer, train_one_epoch
from utils.visuals import show_side_by_side


def train_standard_autoencoder(  # noqa: PLR0913
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


def run(logging_dir: str):
    """Run experiment: standard autoencoder reconstruction."""
    for bs in [32, 64, 128]:
        for lr in [0.1, 0.001, 0.0001]:
            for epochs in [3, 5, 10]:
                model_config = AutoencoderConfig()

                dataset_config = DatasetConfig(
                    dataset_name="cifar10",
                    batch_size=bs,
                    validation_split=0.1,
                )

                train_config = TrainConfig(
                    optimizer="adam",
                    learning_rate=lr,
                    n_epochs=epochs,
                )

                cmlogger = CavemanLogger(root_dir=logging_dir)

                train_standard_autoencoder(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    train_config=train_config,
                    logger=cmlogger,
                    validation_interval=1,
                    test_interval=1,
                    visualization_interval=1,
                )
