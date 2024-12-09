"""Run the whole train/evaluate pipeline."""

from collections.abc import Callable
from dataclasses import dataclass

from torch.optim import Adam, AdamW

from datasets import load_data
from evaluate import evaluate
from models.autoencoder import Autoencoder, AutoencoderConfig
from train import train_one_epoch
from utils.nn import LossFnName, get_loss_fn_by_name


@dataclass
class RunConfig:
    """Config for complete training run."""

    model: str
    model_args: dict
    dataset: str
    batch_size: int
    learning_rate: float
    epochs: int
    loss_fn: LossFnName
    use_weight_decay: bool = False
    reconstruct: bool = False
    validation_interval: int = 1
    callbacks: dict[str, tuple[Callable, int]] | None = None


def run(config: RunConfig) -> None:
    """Run script."""
    # Create train, val, test split
    train_loader, val_loader, test_loader = load_data(
        config.dataset,
        validation_split=0.2,
        batch_size=config.batch_size,
    )

    # Create model and optimizer
    if config.model == "autoencoder":
        model_config = AutoencoderConfig(**config.model_args)
        model = Autoencoder(model_config)
    else:
        msg = "Only autoencoder implemented, sorry."
        raise NotImplementedError(msg)

    optimizer = (
        AdamW(model.parameters(), lr=config.learning_rate)
        if config.use_weight_decay
        else Adam(model.parameters(), lr=config.learning_rate)
    )

    # Training loop
    for epoch in range(config.epochs):
        # Train model for one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=get_loss_fn_by_name(config.loss_fn),
            reconstruct=config.reconstruct,
        )

        # Evaluate model on validation set in specified intervals
        if epoch % config.validation_interval == 0:
            val_loss = evaluate(
                model=model,
                test_loader=test_loader,
                loss_fn=get_loss_fn_by_name(config.loss_fn),
                reconstruct=config.reconstruct,
            )

            print(f"Epoch {epoch}")
            print(f" Train loss {train_loss:.4f}")
            print(f" Val loss {val_loss:.4f}")

        # Call side functions in specified intervals
        if (
            config.callbacks is not None
            and config.callbacks_interval % epoch == 0
        ):
            msg = "Feature not yet implemented sorry hehe"
            raise NotImplementedError(msg)
