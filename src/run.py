"""Run script."""

from datasets import DatasetConfig
from logger.caveman import CavemanLogger
from models.autoencoder import AutoencoderConfig, run
from utils.train import TrainConfig

model_config = AutoencoderConfig()

dataset_config = DatasetConfig(
    dataset_name="cifar10",
    batch_size=32,
    validation_split=0.1,
)

train_config = TrainConfig(
    optimizer="adam",
    learning_rate=0.001,
    n_epochs=20,
)

logger = CavemanLogger(root_dir="testrun")

run(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    logger=logger,
    validation_interval=1,
    test_interval=1,
    visualization_interval=1,
)
