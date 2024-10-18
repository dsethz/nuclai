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
import ipdb

import lightning as L

from datetime import date
from generative.networks.nets import VQVAE
from nuclai.utils.datamodule import DataModule


def _get_args(mode: str) -> argparse.Namespace:
    """
    This function sets up input arguments for train/test/predict.
    """
    # check mode
    assert mode in ["train", "test", "predict"], f"{mode} is an invalid mode."

    # get user input
    parser = argparse.ArgumentParser()

    if mode == 'train':
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
            default=[32, 160, 170],
            help="Shape [depth, heigth, width] that all images will be cropped/padded to before model submission. Default is [32, 160, 170].",
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

    elif mode in ['test', 'predict']:
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


def train():
    """
    This function coordinates model training.
    """
    # get input arguments
    args = _get_args(mode='train')

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

    assert os.path.isfile(path_data_val), f"File {path_data_val} does not exist."

    assert model_type in ["vqvae"], f"Model type {model_type} is not supported."

    assert path_checkpoint is None or os.path.isfile(path_checkpoint), f"File {path_checkpoint} does not exist."

    assert type(epochs) == int and epochs > 0, f"Epochs must be a positive integer."

    assert type(batch_size) == int and batch_size > 0, f"Batch size must be a positive integer."

    assert type(lr) == float and lr > 0, f"Learning rate must be a positive float."

    assert type(shape) == list and len(shape) == 3, f"Shape must be a list of 3 integers."

    assert type(log_frequency) == int and log_frequency > 0, f"Log frequency must be a positive integer."

    assert type(multiprocessing) == bool, f"Multiprocessing must be a boolean."

    assert type(retrain) == bool, f"Retrain must be a boolean."

    assert seed is None or (type(seed) == int and seed > 0), f"Seed must be None or a positive integer."

    assert type(output_base_dir) == str, f"Output base directory must be a string."

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
        gpus = None
        strategy = None
        sync_batchnorm = False
        num_processes = 1
    else:
        accelerator = "gpu"
        gpus = [int(device) for device in devices]
        num_processes = len(gpus)

    # assert correct setup for multiprocessing
    if multiprocessing:
        assert (
            accelerator == "gpu"
        ), "multiprocessing is only enabled for GPU devices."
        assert (
            len(gpus) > 1
        ), f"multiprocessing requires >1 devices, but {len(gpus)} devices are provided."

        # NOTE: currently only single node training supported (else -> batch_size / (ngpus * nnodes))
        batch_size = int(batch_size / len(gpus))
        # sync_batchnorm = True not needed for VQVAE as it uses layer_norm
        strategy = "ddp"
    elif accelerator == "gpu":
        gpus = 1
        strategy = None
        num_processes = 1
        # sync_batchnorm = False not needed for VQVAE as it uses layer_norm
    
    # set up data
    data_module = DataModule(
            path_data=path_data,
            path_data_val=path_data_val,
            batch_size=batch_size,
            shape=tuple(shape),
        )

    ipdb.set_trace()

    # random seeding
    if seed is not None:
        L.pytorch.seed_everything(seed, workers=True)

    # set up model


def test():
    """
    This function coordinates model testing.
    """
    pass


def predict():
    """
    This function coordinates model prediction.
    """
    pass
