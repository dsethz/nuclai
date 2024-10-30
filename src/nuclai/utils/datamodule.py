########################################################################################################################
# Contains the data set classes for nucleus representation learning and classification with a CNN.                     #
# Author:               Daniel Schirmacher                                                                             #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.12.7                                                                                         #
# PyTorch Version:      2.4.1                                                                                          #
# Lightning Version:    2.4.0                                                                                          #
########################################################################################################################
import json
import pathlib
from os import path
from typing import Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import tifffile
import torch
from monai import transforms
from torch.utils.data import DataLoader

from nuclai.utils.utils import _get_mask


class DataSetCls:
    """
    Dataset class for nucleus classification.

    Args:
        path_data: path to JSON file containing feature matrix paths and labels.
        mode: 'train', 'test' or 'predict'. Default is 'train'.
    """

    def __init__(
        self,
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        mode: str = "train",
    ):
        super().__init__()

        # assert input
        assert type(path_data) in (
            str,
            pathlib.PosixPath,
            pathlib.WindowsPath,
        ), f'path_data should be of type "str"/"pathlib.PosixPath"/"pathlib.WindowsPath" but is of type "{type(path_data)}".'

        assert path.exists(
            path_data
        ), f'path_data does not exist, you typed: "{path_data}".'

        assert mode in [
            "train",
            "test",
            "predict",
        ], f'mode should be in ["train", "test", "predict"] but is "{mode}".'

        self.mode = mode
        self.path_data = path_data
        with open(self.path_data) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        read and return item

        Args:
            idx : int
                sample index.

        Returns
        -------
        features : torch.tensor
            Features as torch tensor.
        idx : int
            index to image.

        """
        path_features = self.data[str(idx)]["path"]
        features = np.load(path_features)
        features_t = torch.from_numpy(features).type(torch.FloatTensor)
        features_t = torch.squeeze(features_t)

        if self.mode != "predict":
            label = self.data[str(idx)]["label"]
            label_t = torch.tensor([label], dtype=torch.float32)

            return features_t, label_t, idx

        return features_t, idx


class DataModuleCls(L.LightningDataModule):
    """
    Pytorch lightning class which encapsulates DataSet.

    Args:
        path_data: path to CSV file containing image paths and header "image". Required for training/testing/predicting.
        path_data_val: path to CSV file containing image paths and header "image". Only required for training.
        batch_size: batch size for training.
    """

    def __init__(
        self,
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        path_data_val: Optional[
            Union[str, pathlib.PosixPath, pathlib.WindowsPath]
        ] = None,
        batch_size: int = 2,
    ):
        super().__init__()

        self.path_data = path_data
        self.path_data_val = path_data_val
        self.batch_size = batch_size

        # get feature_dim from first sample
        with open(self.path_data) as f:
            data = json.load(f)

        features = np.load(data["0"]["path"])
        self.feature_dim = features.shape[-1]

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate datasets
        """
        if stage == "fit" or stage is None:
            assert self.path_data_val is not None, "path_data_val is missing."

            self.data = DataSetCls(
                self.path_data,
                mode="train",
            )
            self.data_val = DataSetCls(
                self.path_data_val,
                mode="train",
            )

        if stage == "test" or stage is None:
            self.data_test = DataSetCls(
                self.path_data,
                mode="test",
            )

        if stage == "predict" or stage is None:
            self.data_predict = DataSetCls(
                self.path_data,
                mode="predict",
            )

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


class DataSet:
    """
    Dataset class for nucleus representation learning and classification with a CNN.

    Args:
        path_data: path to CSV file containing image paths and header "image".
        trans: Compose of transforms to apply to each image.
        shape: shape of the input image.
        bit_depth: bit depth of the input image.
    """

    def __init__(
        self,
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        trans: Optional[transforms.Compose] = None,
        shape: tuple[int, ...] = (30, 300, 300),
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
        if trans is not None:
            assert isinstance(
                trans, transforms.Compose
            ), f'trans should be of type "monai.transforms.Compose" but is of type "{type(trans)}".'

        # assert shape
        assert isinstance(
            shape, tuple
        ), f'type of shape should be tuple instead it is of type: "{type(shape)}".'

        assert all(
            isinstance(i, int) for i in shape
        ), "values of shape should be of type integer."

        assert isinstance(
            bit_depth, int
        ), f'type of bit_depth should be int instead it is of type: "{type(bit_depth)}".'

        self.path_data = path_data
        self.data = pd.read_csv(path_data)
        self.shape = shape
        self.trans = trans
        self.padder = transforms.SpatialPad(self.shape, method="symmetric")

        assert (
            "image" in self.data.columns
        ), 'The input file requires "image" as header.'

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        read data (csv file)

        Args:
            idx : int
                index to image.

        Returns:
            image : torch.tensor
                preprocessed image.
            idx : int
                index to image.
        """
        img_path = self.data.loc[idx, "image"]
        img = tifffile.imread(img_path)

        img, mask = self._preprocess(img)

        return img, mask, idx

    def _preprocess(self, img: np.array) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise, augment and transform image.


        Args:
            img : np.array
                input image (z, y, x).

        Returns:
            img_trans : torch.tensor
                transformed img (channel, depth, height, width).
        """

        assert (
            len(img.shape) == 3
        ), f'images are expected to be grayscale and len(img.shape)==3, here it is: "{len(img.shape)}".'

        img = img.astype(self.bit_depth)

        img_t = torch.from_numpy(img).type(torch.FloatTensor)
        img_t = torch.unsqueeze(img_t, 0)

        # apply transforms
        if self.trans is not None:
            img_t = self.trans(img_t)

        # TODO: for now apply padding after transformations (see comment in DataModule.setup)
        img_t = self.padder(img_t)
        mask = _get_mask(img=img_t, padder=self.padder)

        return img_t, mask


class DataModule(L.LightningDataModule):
    """
    Pytorch lightning class which encapsulates DataSet.

    Args:
        path_data: path to CSV file containing image paths and header "image". Required for training/testing/predicting.
        path_data_val: path to CSV file containing image paths and header "image". Only required for training.
        batch_size: batch size for training.
        shape: shape of the input image.
    """

    def __init__(
        self,
        path_data: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        path_data_val: Optional[
            Union[str, pathlib.PosixPath, pathlib.WindowsPath]
        ] = None,
        batch_size: int = 2,
        shape: tuple[int, ...] = (30, 300, 300),
    ):
        super().__init__()

        self.path_data = path_data
        self.path_data_val = path_data_val
        self.batch_size = batch_size
        self.shape = shape

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate datasets
        """

        # catch image data type
        tmp = pd.read_csv(self.path_data)
        img = tifffile.imread(tmp.loc[0, "image"])

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
            assert self.path_data_val is not None, "path_data_val is missing."

            # instantiate transforms and datasetst
            # TODO: use translation to not only have centered images (e.g. RandAffine)
            trans = transforms.Compose(
                [
                    transforms.NormalizeIntensity(
                        subtrahend=0, divisor=max_intensity
                    ),
                    transforms.RandZoom(keep_size=True),
                    transforms.RandAxisFlip(),
                    transforms.RandAdjustContrast(),
                    transforms.RandCoarseDropout(
                        holes=300, spatial_size=(2, 10, 10), fill_value=0.0
                    ),  # 300 holes cover ~2% of padded images --> good chance to be visible without blocking small cells
                ]
            )

            trans_val = transforms.Compose(
                [
                    transforms.NormalizeIntensity(
                        subtrahend=0, divisor=max_intensity
                    ),
                ]
            )

            self.data = DataSet(
                self.path_data,
                trans=trans,
                shape=self.shape,
                bit_depth=bit_depth,
            )
            self.data_val = DataSet(
                self.path_data_val,
                trans=trans_val,
                shape=self.shape,
                bit_depth=bit_depth,
            )

        if stage == "test" or stage is None:
            # instantiate transforms and datasets
            trans = transforms.Compose(
                [
                    transforms.NormalizeIntensity(
                        subtrahend=0, divisor=max_intensity
                    ),
                ]
            )

            self.data_test = DataSet(
                self.path_data,
                trans=trans,
                shape=self.shape,
                bit_depth=bit_depth,
            )

        if stage == "predict" or stage is None:
            # instantiate transforms and datasets
            trans = transforms.Compose(
                [
                    transforms.NormalizeIntensity(
                        subtrahend=0, divisor=max_intensity
                    ),
                ]
            )

            self.data_predict = DataSet(
                self.path_data,
                trans=trans,
                shape=self.shape,
                bit_depth=bit_depth,
            )

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
