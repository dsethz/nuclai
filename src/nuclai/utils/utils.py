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

import numpy as np
import torch
from monai.data import MetaTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import CropForegroundd, SpatialPad
from monai.transforms.croppad.functional import pad_func
from monai.utils import TraceKeys, convert_to_tensor, fall_back_tuple
from skimage import io


class RandomSpatialPad(SpatialPad):
    """
    Randomly pad input along the last n axes given a defined n-dimensional shape.

    Args:
        spatial_size: Union[int, tuple[int, ...], list[int, ...]]
            Expected shape of spatial dimensions after padding.
        **kwargs: Any
            Additional parameters for parent class.

    TODO:
    * adapt DataModule to use RandomSpatialPad
        * add to compose of each data set

    """

    def __init__(
        self,
        spatial_size: Union[int, tuple[int, ...], list[int, ...]],
        **kwargs,
    ):
        super().__init__(spatial_size=spatial_size, **kwargs)

    def __call__(self, img: MetaTensor) -> MetaTensor:
        # get img shape
        input_spatial_shape = img.shape[1:]  # assume channel first format

        # compute padding
        pads = self._compute_pad_width(input_spatial_shape)

        # Convert img to metatensor if necessary
        img = convert_to_tensor(data=img, track_meta=get_track_meta())

        return pad_func(img, pads, self.get_transform_info())

    def _compute_pad_width(self, input_spatial_shape):
        # validate spatial_size
        spatial_size = fall_back_tuple(self.spatial_size, input_spatial_shape)

        # compute padding
        pads = []
        for dim_size, target_size in zip(
            input_spatial_shape, spatial_size, strict=False
        ):
            total_pad = max(target_size - dim_size, 0)
            pad_before = np.random.randint(0, total_pad + 1)
            pad_after = total_pad - pad_before
            pads.append((pad_before, pad_after))

        # add channel dimension
        pads = tuple([(0, 0)] + pads)

        return pads


def _threshold_at_zero(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function for crop_original.
    """
    # threshold at 0
    return x > 0


def _min_max_normalize(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image to [0, 1].

    Args:
        img: torch.Tensor
            Image tensor of shape (C, D, H, W).

    Returns:
        torch.Tensor
            Normalized image tensor of shape (C, D, H, W).
    """
    return (img - img.min()) / (img.max() - img.min())


def crop_original(img: MetaTensor, mask: MetaTensor) -> MetaTensor:
    """
    Crop the original image to the bounding box defined by the mask.

    Args:
        img: MetaTensor
            Image tensor of shape (C, D, H, W).
        mask: MetaTensor
            Mask tensor of shape (C, D, H, W).

    Returns:
        MetaTensor
            Cropped image tensor of shape (C, D, H, W).
    """
    # TODO: consider moving this to DataSet.
    # create helper dict
    tmp = {"image": img, "mask": mask}

    # create cropper
    cropper = CropForegroundd(
        select_fn=_threshold_at_zero, keys="image", source_key="mask"
    )

    return cropper(tmp)["image"]


def save_image_mod(
    img: torch.Tensor,
    fp: Union[str, pathlib.Path, BinaryIO],
    znorm: bool = False,
) -> None:
    """
    torchvision.utils.save_image modified to save gray_scale images.

    Args:
        img: torch.Tensor
            Image to be saved.
        fp: string or file object
            A filename or a file object.
        znorm: bool
            If True, normalize the image to [0, 1] before saving.
    """
    assert len(img.shape) in (
        4,
        5,
    ), "img.size() is expected to be of len 4 or 5, but is of len {len(img.size())}."

    if len(img.size()) == 5:
        img[0, :, :, :, :]

    n_channels = img.size()[0]

    # normalize to [0, 1]
    if znorm:
        img = _min_max_normalize(img)

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


def _get_mask(img: MetaTensor, padder: SpatialPad) -> MetaTensor:
    """
    Compute mask tensor from img.
    ATTENTION: This only works if padding was the last transform used on img.
    See https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/transforms/croppad/array.py#L170

    Args:
        img: MetaTensor
            Image tensor of shape (C, D, H, W).
        padder: SpatialPad
            The SpatialPad object that was used to obtain (C, D, H, W) shaped Image.

    Returns:
        MetaTensor
            The mask tensor of shape (C, D, H, W).
    """
    transform = padder.pop_transform(
        img
    )  # ATTENTION: pop_transform removes transform from list?
    padded = transform[TraceKeys.EXTRA_INFO]["padded"]
    if padded[0][0] > 0 or padded[0][1] > 0:  # slicing the channel dimension
        s = padded[0][0]
        e = min(max(padded[0][1], s + 1), len(img))
        img = img[s : len(img) - e]  # type: ignore
    roi_start = [i[0] for i in padded[1:]]
    roi_end = [
        i - j[1] for i, j in zip(img.shape[1:], padded[1:], strict=False)
    ]

    mask = torch.zeros_like(img)
    mask[
        :,
        roi_start[0] : roi_end[0],
        roi_start[1] : roi_end[1],
        roi_start[2] : roi_end[2],
    ] = 1

    return mask
