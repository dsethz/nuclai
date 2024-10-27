########################################################################################################################
# This script contains training, testing, and predicting entry points for representation learning.                     #
# Author:               Daniel Schirmacher                                                                             #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.12.7                                                                                         #
# PyTorch Version:      2.4.1                                                                                          #
# Lightning Version:    2.4.0                                                                                          #
########################################################################################################################
# Imports
import argparse
import os
import random
import re
from datetime import date

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner

from nuclai.models.vqvae import LitVQVAE
from nuclai.utils.callbacks import CheckpointCallback
from nuclai.utils.datamodule import DataModule


def _get_args(mode: str) -> argparse.Namespace:
    """
    This function sets up input arguments for train/test/predict.
    """
    # check mode
    assert mode in ["train", "test", "predict"], f"{mode} is an invalid mode."

    # get user input
    parser = argparse.ArgumentParser()

    if mode == "train":
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            help="Path to CSV file containing training image file paths.",
        )

        parser.add_argument(
            "--data_val",
            type=str,
            required=True,
            help="Path to CSV file containing validation image file paths.",
        )

        parser.add_argument(
            "--model",
            type=str,
            default="vqvae",
            help="Model type to train. Default is vqvae",
        )

        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to checkpoint file of trained pl.LightningModule. Default is None.",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=5,
            help="Number of training epochs. Default is 5.",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=2,
            help="Number of samples per mini-batch. Default is 2.",
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Learning rate of the optimizer. Default is 1e-4.",
        )

        parser.add_argument(
            "--shape",
            type=int,
            nargs="+",
            default=[40, 336, 224],
            help="Shape [depth, heigth, width] that all images will be cropped/padded to "
            "before model submission. Along all axes shape evenly divisible "
            "by 2**len(downsample_parameters) MUST hold. Default is [32, 200, 200].",
        )

        parser.add_argument(
            "--log_frequency",
            type=int,
            default=50,
            help="Log performance metrics every N gradient steps during training. Default is 50.",
        )

        # set default to false
        parser.add_argument(
            "--multiprocessing",
            action="store_true",
            help="If flag is used, all GPUs given in devices will be used for traininig/inference. Does not support CPU.",
        )

        parser.add_argument(
            "--retrain",
            action="store_true",
            help="If flag is used, best scores for model saving will be reset (required for training on new data).",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="None or Int to use for random seeding. Default is None.",
        )

    elif mode in ["test", "predict"]:
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            help="Path to CSV file containing image file paths.",
        )

        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to checkpoint file of trained pl.LightningModule.",
        )

        parser.add_argument(
            "--suffix",
            type=str,
            default="",
            help="Suffix to append to all mask file names.",
        )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu"],
        help='Devices to use. Can be numerical GPU IDs or "cpu". Default is "cpu".',
    )

    return parser.parse_args()


def _initialise_inferrence(
    data: str,
    model: str,
    devices: list[str],
    output_base_dir: str,
    suffix: str,
) -> tuple[L.Trainer, L.LightningModule, L.LightningDataModule]:
    """
    Construct trainer, model, and data module for testing/predicting
    """
    # check input arguments
    assert os.path.isfile(data), f"File {data} does not exist."

    assert os.path.isfile(model), f"File {model} does not exist."

    assert isinstance(devices, list), "Devices must be a list."

    assert isinstance(
        output_base_dir, str
    ), "Output base directory must be a string."

    assert isinstance(suffix, str), "Suffix must be a string."

    # create directories
    os.makedirs(output_base_dir, exist_ok=True)

    # ensure compatibility of devices with $CUDA_VISIBLE_DEVICES input
    if len(devices) == 1 and "," in devices[0]:
        devices = devices[0].split(",")

    if "cpu" in devices:
        accelerator = "cpu"
        n_devices = 1
    else:
        accelerator = "gpu"
        n_devices = 1  # test only on one gpu

    # load model
    if os.path.isfile(model):
        model = LitVQVAE.load_from_checkpoint(model)
        model.suffix = suffix
    else:
        raise FileNotFoundError(f'The file "{model}" does not exist.')

    # set up data
    data_module = DataModule(
        path_data=data,
        batch_size=1,
        shape=model.shape,
    )

    # test model
    logger = CSVLogger(output_base_dir, name="lightning_logs")
    trainer = L.Trainer(
        default_root_dir=output_base_dir,
        accelerator=accelerator,
        devices=n_devices,
        logger=logger,
    )

    return trainer, model, data_module


def train():
    """
    This function coordinates model training.
    """
    # get input arguments
    args = _get_args(mode="train")

    path_data = args.data
    path_data_val = args.data_val
    model_type = args.model
    path_checkpoint = args.checkpoint
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    shape = args.shape
    log_frequency = args.log_frequency
    multiprocessing = args.multiprocessing
    retrain = args.retrain
    seed = args.seed
    output_base_dir = args.output_base_dir
    devices = args.devices

    # check input arguments
    assert os.path.isfile(path_data), f"File {path_data} does not exist."

    assert os.path.isfile(
        path_data_val
    ), f"File {path_data_val} does not exist."

    assert model_type in [
        "vqvae"
    ], f"Model type {model_type} is not supported."

    assert path_checkpoint is None or os.path.isfile(
        path_checkpoint
    ), f"File {path_checkpoint} does not exist."

    assert (
        isinstance(epochs, int) and epochs > 0
    ), "Epochs must be a positive integer."

    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), "Batch size must be a positive integer."

    assert (
        isinstance(lr, float) and lr > 0
    ), "Learning rate must be a positive float."

    assert (
        isinstance(shape, list) and len(shape) == 3
    ), "Shape must be a list of 3 integers."

    assert (
        isinstance(log_frequency, int) and log_frequency > 0
    ), "Log frequency must be a positive integer."

    assert isinstance(
        multiprocessing, bool
    ), "Multiprocessing must be a boolean."

    assert isinstance(retrain, bool), "Retrain must be a boolean."

    assert seed is None or (
        isinstance(seed, int) and seed > 0
    ), "Seed must be None or a positive integer."

    assert isinstance(
        output_base_dir, str
    ), "Output base directory must be a string."

    # reformat shape to tuple
    shape = tuple(shape)

    # create directories
    d = date.today()
    identifier = (
        str(d.year)[2:]
        + str(d.month).zfill(2)
        + str(d.day).zfill(2)
        + "_"
        + model_type
    )
    rnd_id = str(random.getrandbits(15)).zfill(5)

    while os.path.exists(
        os.path.join(args.output_base_dir, f"{identifier}_{rnd_id}")
    ):
        rnd_id = str(random.getrandbits(15)).zfill(5)

    identifier += f"_{rnd_id}"

    output_base_dir = os.path.join(output_base_dir, identifier)
    os.makedirs(output_base_dir)

    del d, rnd_id

    # ensure compatibility of devices with #CUDA_VISIBLE_DEVICES
    if len(devices) == 1 and "," in devices[0]:
        devices = devices[0].split(",")

    if "cpu" in devices:
        accelerator = "cpu"
        strategy = "auto"
        n_devices = 1
        precision = "32-true"
    else:
        accelerator = "gpu"
        n_devices = len([int(device) for device in devices])
        precision = "16-mixed"

    # assert correct setup for multiprocessing
    if multiprocessing:
        assert (
            accelerator == "gpu"
        ), "multiprocessing is only enabled for GPU devices."
        assert (
            n_devices > 1
        ), f"multiprocessing requires >1 devices, but {n_devices} devices are provided."

        batch_size = int(batch_size / n_devices)
        strategy = "ddp"
    elif accelerator == "gpu":
        strategy = "auto"
        n_devices = 1

    # set up data
    data_module = DataModule(
        path_data=path_data,
        path_data_val=path_data_val,
        batch_size=batch_size,
        shape=shape,
    )

    # random seeding
    if seed is not None:
        L.pytorch.seed_everything(seed, workers=True)

    # set up model
    if model_type == "vqvae":
        model = LitVQVAE(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(256, 256, 256),
            num_res_channels=256,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=(
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
                (2, 4, 1, 1, 0),
            ),
            num_embeddings=256,
            embedding_dim=32,
            embedding_init="normal",
            commitment_cost=0.25,
            decay=0.5,
            epsilon=1e-5,
            dropout=0.0,
            ddp_sync=True,
            use_checkpointing=False,
            shape=shape,
            learning_rate=lr,
            suffix="",
        )
    else:
        raise ValueError(f"Model type {model_type} is not supported.")

    # set up callback for best model
    checkpoint_best_loss = ModelCheckpoint(
        monitor="loss_val",
        filename="best-loss-{epoch}-{step}",
        mode="min",
    )

    checkpoint_latest = ModelCheckpoint(
        monitor=None,
        filename="latest-{epoch}-{step}",
        mode="max",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # update max_epoch when loading from checkpoint
    if path_checkpoint is not None:
        epoch_pattern = re.compile(r"epoch=([0-9]+)")
        old_epoch = int(epoch_pattern.search(path_checkpoint)[1])
        epochs += old_epoch

    # set up trainer
    logger = CSVLogger(output_base_dir, name="lightning_logs")
    trainer = L.Trainer(
        max_epochs=epochs,
        default_root_dir=output_base_dir,
        accelerator=accelerator,
        strategy=strategy,
        devices=n_devices,
        logger=logger,
        callbacks=[
            checkpoint_best_loss,
            checkpoint_latest,
            lr_monitor,
            CheckpointCallback(retrain=retrain),
        ],
        log_every_n_steps=log_frequency,
        precision=precision,
    )

    # find optimal learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=data_module)

    lr_finder.plot(suggest=True)
    plt.savefig(os.path.join(output_base_dir, "lr_finder.png"))
    plt.close()

    new_lr = lr_finder.suggestion()

    # only adapt lr if new_lr too small
    # we use OnPlateau scheduler and don't want to start with too low lr
    if new_lr >= 1e-5:
        model.hparams.learning_rate = new_lr
        model.learning_rate = new_lr  # not sure if both necessary
    else:
        model.hparams.learning_rate = lr
        model.learning_rate = lr

    # train model
    trainer.fit(model, data_module, ckpt_path=path_checkpoint)


def test():
    """
    This function coordinates model testing.
    """
    # get input arguments
    args = _get_args(mode="test")
    trainer, model, data_module = _initialise_inferrence(
        data=args.data,
        model=args.model,
        devices=args.devices,
        output_base_dir=args.output_base_dir,
        suffix=args.suffix,
    )
    trainer.test(model, data_module)


def predict():
    """
    This function coordinates model prediction.
    """
    # get input arguments
    args = _get_args(mode="predict")
    trainer, model, data_module = _initialise_inferrence(
        data=args.data,
        model=args.model,
        devices=args.devices,
        output_base_dir=args.output_base_dir,
        suffix=args.suffix,
    )
    trainer.predict(model, data_module)
