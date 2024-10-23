#######################################################################################################################
# This script contains utility functions.                                                                             #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.12.7                                                                                        #
# PyTorch Version:      2.4.1                                                                                         #
# Torchvision  Version: 0.19.1                                                                                        #
#######################################################################################################################

import pathlib
from typing import BinaryIO, Union

import torch
from skimage import io


def save_image_mod(
    img: torch.Tensor,
    fp: Union[str, pathlib.Path, BinaryIO],
) -> None:
    """
    torchvision.utils.save_image modified to save gray_scale images.

    Args:
        img: torch.Tensor
            Image to be saved.
        fp: string or file object
            A filename or a file object.
    """
    assert len(img.shape) in (
        4,
        5,
    ), "img.size() is expected to be of len 4 or 5, but is of len {len(img.size())}."

    if len(img.size()) == 5:
        img[0, :, :, :, :]

    n_channels = img.size()[0]

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer --> modified to obtain grayscale image
    if n_channels == 1:
        ndarr = (
            img[0]
            .mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .to("cpu", torch.uint8)
            .numpy()
        )
    else:
        ndarr = (
            img.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 3, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )

    io.imsave(fp, ndarr)
