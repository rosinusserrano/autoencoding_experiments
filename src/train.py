"""Run script."""

import argparse

import mlflow

from datasets import DatasetConfig
from logger.mlflow import MlFlowLogger
from models.autoencoder import AutoencoderV2Config, AutoencoderConfig, mse_loss
from models.vae import VAEConfig, mse_and_kld_loss
from utils.train import TrainConfig, standard_training_pipeline
from utils.parse import config_to_parser


main_parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

main_parser.add_argument(
    "--experiment",
    "-e",
    help="The name of the experiment.",
)

# TRAIN CONFIG

train_parser = main_parser.add_argument_group(
    title="Train config",
    description="Arguments configuring training behaviour",
)
config_to_parser(train_parser, TrainConfig)

# DATASET CONFIG

dataset_parser = main_parser.add_argument_group(
    title="Dataset config",
    description="Arguments configuring the dataset and -loader",
)
config_to_parser(dataset_parser, DatasetConfig)

# MODEL CONFIGS

model_subparsers = main_parser.add_subparsers(
    help="The model to train",
    title="Model config",
    description="This is the model you will ultimately train",
    dest="model",
)

ae_parser = model_subparsers.add_parser(
    "ae",
    help="Standard, fully convolutional autoencoder",
    description="Standard, fully convolutional autoencoder",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
config_to_parser(ae_parser, AutoencoderConfig)

aev2_parser = model_subparsers.add_parser(
    "aev2",
    help="Autoencoder (v2) with a fully-connected bottleneck",
    description="Autoencoder with fully connected bottleneck",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
config_to_parser(aev2_parser, AutoencoderV2Config)

vae_parser = model_subparsers.add_parser(
    "vae",
    help="VAE, fully convolutional",
    description="VAE, fully convolutional",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
config_to_parser(vae_parser, VAEConfig)

args = main_parser.parse_args()

print(args)


model_config = AutoencoderV2Config(
    encoder_fc_features=[9216, 9216, 8192],
    decoder_fc_features=[8192, 9216, 9216],
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
