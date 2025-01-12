"""Run script."""

import argparse

import mlflow

from datasets import DatasetConfig
from logger.caveman import CavemanLogger
from logger.mlflow import MlFlowLogger
from models.autoencoder import AutoencoderV2Config, mse_loss
from models.vae import VAEConfig, mse_and_kld_loss
from utils.train import TrainConfig, standard_training_pipeline


# parser = argparse.ArgumentParser(prog="aexp-run")

# parser.add_argument("--model-name", "-m", )


model_config = AutoencoderV2Config(
    encoder_fc_features=[9216, 7680, 4096],
    decoder_fc_features=[4096, 7680, 9216],
)

dataset_config = DatasetConfig(
    root="/workspace/drive",
    dataset_name="stl10",
    validation_split=0.1,
    batch_size=128,
)
train_config = TrainConfig(
    "adam",
    learning_rate=0.0005,
    n_epochs=20,
    weight_decay=0.0001,
)

experiment_name = "AutoencoderV2 on STL10"

mlflow.end_run()
logger = MlFlowLogger(
    experiment_name=experiment_name,
    remote_url="https://mlflow.sniggles.de",
    debug=True,
)

standard_training_pipeline(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    logger=logger,
    loss_fn=mse_loss,
    validation_interval=1,
    test_interval=1,
    visualization_interval=1,
)
