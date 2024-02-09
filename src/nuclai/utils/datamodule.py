########################################################################################################################
# This script handels the loading and processing of the input dataset for cell classification with a CNN.              #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# PyTorch Version:      2.1.2                                                                                          #
# Lightning Version:    2.1.3                                                                                          #
########################################################################################################################
import pathlib
import random
from os import path
from typing import BinaryIO, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pandas as pd
import pytorch_lightning as pl
import torch
import math
from PIL import Image
from skimage import io
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


class Datasets:

    def __init__(
        self, 
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        transform_both: Optional[transforms.transforms.Compose] = None,
        transform_img: Optional[transforms.transforms.Compose] = None,
        transform_mask: Optional[transforms.transforms.Compose] = None,
        shape: Tuple[int, int] = (30, 300, 300),
        bit_depth: int = 8,
    ):

        super().__init__()

        # assert input path
        assert type(path_data) in (
            str,
            pathlib.PosixPath,
            pathlib.WindowsPath,
        ), f'path_data should be of type "str"/"pathlib.PosixPath"/"pathlib.WindowsPath" but is of type "{type(path_data)}".'

        assert path.exists(
            path_data
        ), f'path_data does not exist, you typed: "{path_data}".'

        # transformation
        if transform_both is not None:
            assert (
                type(transform_both) == transforms.transforms.Compose
            ), f'transform_both should be of type "torchvision.transforms.transforms.Compose" but is of type "{type(transform_both)}".'

        if transform_img is not None:
            assert (
                type(transform_img) == transforms.transforms.Compose
            ), f'transform_img should be of type "torchvision.transforms.transforms.Compose" but is of type "{type(transform_img)}".'

        if transform_mask is not None:
            assert (
                type(transform_mask) == transforms.transforms.Compose
            ), f'transform_mask should be of type "torchvision.transforms.transforms.Compose" but is of type "{type(transform_mask)}".'

        # assert shape
        assert (
            type(shape) == tuple
        ), f'type of shape should be tuple instead it is of type: "{type(shape)}".'

        assert all(
            isinstance(i, int) for i in shape
        ), "values of shape should be of type integer."

        assert (
            type(bit_depth) == int
        ), f'type of bit_depth should be int instead it is of type: "{type(bit_depth)}".'

        self.path_data = path_data
        self.data = pd.read_csv(path_data)
        self.shape = shape
        self.transform_both = transform_both
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self._padder = transforms.RandomCrop(self.shape, pad_if_needed=True)

        assert all(
            col in self.data.columns for col in ("DAPI", "mask", "label_1")
        ), 'The input file requires ("DAPI", "mask", "label_1") as headers.'

        if bit_depth == 8:
            self.bit_depth = np.uint8
        elif bit_depth == 16:
            self.bit_depth = np.int32
        else:
            self.bit_depth = np.uint8
            raise Warning(
                f'bit_depth must be in {8, 16}, but is "{bit_depth}". It will be handled as 8bit and may create an integer overflow.'
            )
        

        def __len__(self):
            return len(self.data)
        

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]: 
            """
            read data (csv file)

            Parameters
            ----------
            idx : int
                index to image.

            Returns
            -------
            image : torch.tensor
                preprocessed image.
            mask : torch.tensor
                preprocessed mask.
            idx : int
                index to image.

            """
            image_path = self.data.loc[idx, 'DAPI']
            image = io.imread(image_path, as_gray=True)
            image = img_as_ubyte(image) #int byte [0, 255]

            mask_path = self.data.loc[idx, 'mask']
            mask = io.imread(mask_path, as_gray=True)
            mask = np.array(mask) #circumvent read only arrays
            mask[mask > 0] = 255
            mask = mask.astype(np.uint8)

            label = self.data.loc[idx, 'label_1']
            label = label.astype(np.float)

            image, mask = self._preprocess(image, mask)

            return image, mask, label, idx



        def _preprocess(self, image: np.array, mask: np.array) -> torch.Tensor:
            """
            Normalise, augment and transform image and mask


            Parameters
            ----------
            image : np.array
                input images (z, y, x).
            mask : np.array
                input masks (z, y, x).

            Returns
            -------
            image_trans : torch.tensor
                transformed image (channel, height, width).
            mask_trans : torch.tensor
                transformed mask (channel, height, width).

            """

            assert (
            len(image.shape) == 3
            ), f'images are expected to be grayscale and len(image.shape)==3, here it is: "{len(image.shape)}".'
            assert (
            len(mask.shape) == 3
            ), f'masks are expected to be grayscale and len(mask.shape)==3, here it is: "{len(mask.shape)}".'

            image = image.astype(self.bit_depth)

            image_trans = torch.from_numpy(image).type(torch.FloatTensor)
            mask_trans = torch.from_numpy(mask).type(torch.FloatTensor)

            # apply self.transform_img
            if self.transform_image is not None:
                image_trans = self.transform_image(image_trans)

            # apply self.transform_mask
            if self.transform_mask is not None:
                mask_trans = self.transform_mask(mask_trans)

            # to apply the same augmentations on mask and image first merge them
            if self.transform_both is not None:
                _, depth, height, width = image_trans.size()
                merged = torch.zeros(2, depth, height, width)
                merged[0, :, :, :] = image_trans
                merged[1, :, :, :] = mask_trans

                merged_trans = self._padder(merged)
                merged_trans = self.transform_both(merged_trans)

                image_trans = merged_trans[0, :, :, :]
                image_trans = image_trans[None, :, :] #is that to drop dimension 1 ?

                mask_trans = merged_trans[1, :, :, :]
                mask_trans = mask_trans[None, :, :, :]

            # merge them for all cases
            image_mask = torch.zeros(2, depth, height, width)
            image_mask[0, :, :, :] = image_trans
            image_mask[1, :, :, :] = mask_trans
 
            return image_trans







class Datamodule(pl.LightningDataModule):
    """""
    Pytorch lightning class which encapsulates the data preprocess and loading steps
    """""

    def __init__(
        self,
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        path_data_val: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        path_data_test: Optional[
            Union[str, pathlib.PosixPath, pathlib.WindowsPath]
        ] = None,
        path_data_predict: Optional[
            Union[str, pathlib.PosixPath, pathlib.WindowsPath] 
        ] = None,
        batch_size: int = 2,
        shape: Tuple[int, int, int] = (30, 300, 300),
        transform_intensity: bool = False
    ):
        
        super().__init__()  #initializes any attributes from the parent class

        self.path_data = path_data
        self.path_data_val = path_data_val
        self.path_data_test = path_data_test
        self.path_data_predict = path_data_predict
        self.batch_size = batch_size
        self.shape = shape
        self.transform_intensity = transform_intensity

    def setup(self):
        """
        Instantiate datasets
        """

        # catch image data type
        tmp = pd.read_csv(self.path_data)
        img = io.imread(tmp.DAPI[0])
        img = img_as_ubyte(img)

        if img.dtype == np.uint8:
            max_intensity = 255.0
            bit_depth = 8
        elif img.dtype == np.uint16:
            max_intensity = 65535.0
            bit_depth = 16
        else:
            max_intensity = 255.0
            bit_depth = 8
            raise Warning(
                f'Image type "{img.dtype}" is currently not supported and will be converted to "uint8".'
            )

        if stage == "fit" or stage is None:
            transform_both = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                ]
            )

            transform_mask = transforms.Compose(
                [
                    transforms.Normalize(0.0, 255.0),
                ]
            )

            if self.transform_intensity:
                transform_img = transforms.Compose(
                    [
                        transforms.Normalize(0.0, max_intensity),
                        transforms.ColorJitter(
                            brightness=0.7,
                            contrast=0.5,
                            saturation=0.5,
                            hue=0.5,
                        ),
                        transforms.GaussianBlur(kernel_size=5),
                        transforms.RandomAdjustSharpness(4, p=0.5),
                    ]
                )
            else:
                transform_img = transforms.Compose(
                    [
                        transforms.Normalize(0.0, max_intensity),
                    ]
                )

            self.data = Dataset(
                self.path_data,
                transform_both=transform_both,
                transform_img=transform_img,
                transform_mask=transform_mask,
                shape=self.shape,
                bit_depth=bit_depth,
            )
            self.data_val = Dataset(
                self.path_data_val,
                transform_both=transform_both,
                transform_img=transform_img,
                transform_mask=transform_mask,
                shape=self.shape,
                bit_depth=bit_depth,
            )

        if stage == "test" or stage is None:
            transform_mask = transforms.Compose(
                [
                    transforms.Normalize(0.0, 255.0),
                ]
            )
            transform_img = transforms.Compose(
                [
                    transforms.Normalize(0.0, max_intensity),
                ]
            )

            if self.path_data_test is not None:
                self.data_test = Dataset_test(
                    self.path_data_test,
                    transform_img=transform_img,
                    transform_mask=transform_mask,
                    bit_depth=bit_depth,
                )
            else:
                raise ValueError("path_data_test is missing")

        if stage == "predict" or stage is None:
            transform_img = transforms.Compose(
                [
                    transforms.Normalize(0.0, max_intensity),
                ]
            )

            if self.path_data_predict is not None:
                self.data_predict = Dataset_predict(
                    self.path_data_predict,
                    transform_img=transform_img,
                    bit_depth=bit_depth,
                )
            else:
                raise ValueError("path_data_predict is missing")
            
    def train_dataloader(self):
        return DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, batch_size=self.batch_size, num_workers=4
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict, batch_size=self.batch_size, num_workers=0
        )


if __name__=='__main__':
    setup()







