########################################################################################################################
# This is a pytorch lightning wrapper around MONAI's VQ-VAE.                                                           #
# Author:               Daniel Schirmacher                                                                             #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.12.7                                                                                         #
# PyTorch Version:      2.4.1                                                                                          #
# Lightning Version:    2.4.0                                                                                          #
########################################################################################################################
# Imports
import torch

import lightning as L

from generative.networks.nets import VQVAE
from monai.networks.layers import Act
from torch import nn
from torch.nn import functional as F


class LitVQVAE(L.LightningModule):
    """
    Set up of the LightningModule for MONAI's VQVAE.

    Args:
        spatial_dims (int): Number of spatial dimensions.
        in_channels (int): Number of input channels (e.g. 1 for grayscale).
        out_channels (int): Number of output channels of the decoder (e.g. 1 for grayscale).
        num_channels (tuple): Number of channels in the encoder and decoder. len(num_channels)
            must match len(downsample_paramters) and len(upsample_parameters).
        num_res_channels (int): Number of channels in the residual blocks.
        num_res_layers (int): Number of sequential residual layers at each level.
        downsample_parameters (tuple): A Tuple of Tuples for defining the downsampling convolutions.
            Each Tuple should hold the following information stride (int), kernel_size (int),
            dilation (int) and padding (int).
        upsample_parameters (tuple): A Tuple of Tuples for defining the upsampling convolutions.
            Each Tuple should hold the following information stride (int), kernel_size (int),
            dilation (int), padding (int), output_padding (int).
        num_embeddings (int): Number of code vectors in the code book.
        embedding_dim (int): Size of each code vector.
        embedding_init (str): Initialisation of the embeddings. Is in ("normal", "kaiming_uniform").
        commitment_cost (float): Scaling factor (beta) of the MSE loss between input and its quantized
            version. Defaults to 0.25.
        decay (float): EMA decay for codebook updates. Defaults to 0.5.
        epsilon (float): epsilon value for EMA update. Defaults to 1e-5.
        dropout (float): Dropout rate.
        act (tuple | str | None): Activation type and arguments.
        output_act (tuple | str | None): Output activation type and arguments.
        ddp_sync (bool): whether to syncronize codebook across processes.
        use_checkpointing (bool): Whether to use activation checkpointing to save memory.
        learning_rate (float): Learning rate.
        suffix (str): Suffix for the model name.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: tuple = (256, 256),
        num_res_channels: int = 256,
        num_res_layers: int = 2,
        downsample_parameters: tuple = ((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: tuple = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings: int = 256,
        embedding_dim: int = 32,
        embedding_init: str = "normal",
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = Act.RELU,
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,
        learning_rate: float = 1e-3,
        suffix: str = "",
        **kwargs,
    ):
        super().__init__()

        # save hyperparameters from __init__ upon checkpoints in hparams.yaml
        self.save_hyperparameters()

        # assert input formats
        assert isinstance(
            spatial_dims, int
        ), f'spatial_dims is expected to be of type "int" but is of type "{type(spatial_dims)}".'
        assert isinstance(
            in_channels, int
        ), f'in_channels is expected to be of type "int" but is of type "{type(in_channels)}".'
        assert isinstance(
            out_channels, int
        ), f'out_channels is expected to be of type "int" but is of type "{type(out_channels)}".'
        assert isinstance(
            num_channels, tuple
        ), f'num_channels is expected to be of type "tuple" but is of type "{type(num_channels)}".'
        assert isinstance(
            num_res_channels, int
        ), f'num_res_channels is expected to be of type "int" but is of type "{type(num_res_channels)}".'
        assert isinstance(
            num_res_layers, int
        ), f'num_res_layers is expected to be of type "int" but is of type "{type(num_res_layers)}".'
        assert isinstance(
            downsample_parameters, tuple
        ), f'downsample_parameters is expected to be of type "tuple" but is of type "{type(downsample_parameters)}".'
        assert isinstance(
            upsample_parameters, tuple
        ), f'upsample_parameters is expected to be of type "tuple" but is of type "{type(upsample_parameters)}".'
        assert isinstance(
            num_embeddings, int
        ), f'num_embeddings is expected to be of type "int" but is of type "{type(num_embeddings)}".'
        assert isinstance(
            embedding_dim, int
        ), f'embedding_dim is expected to be of type "int" but is of type "{type(embedding_dim)}".'
        assert (
            embedding_init in ("normal", "kaiming_uniform")
        ), f'embedding_init is expected to be of type "str" but is of type "{type(embedding_init)}".'
        assert isinstance(
            commitment_cost, float
        ), f'commitment_cost is expected to be of type "float" but is of type "{type(commitment_cost)}".'
        assert isinstance(
            decay, float
        ), f'decay is expected to be of type "float" but is of type "{type(decay)}".'
        assert isinstance(
            epsilon, float
        ), f'epsilon is expected to be of type "float" but is of type "{type(epsilon)}".'
        assert isinstance(
            dropout, float
        ), f'dropout is expected to be of type "float" but is of type "{type(dropout)}".'
        assert isinstance(
                ddp_sync, bool
        ), f'ddp_sync is expected to be of type "bool" but is of type "{type(ddp_sync)}".'
        assert isinstance(
            use_checkpointing, bool
        ), f'use_checkpointing is expected to be of type "bool" but is of type "{type(use_checkpointing)}".'
        assert isinstance(
            learning_rate, float
        ), f'learning_rate is expected to be of type "float" but is of type "{type(learning_rate)}".'
        assert isinstance(
            suffix, str
        ), f'suffix is expected to be of type "str" but is of type "{type(suffix)}".'

        # additional assertions
        assert len(num_channels) == len(downsample_parameters) == len(upsample_parameters), (
            f"Length of num_channels ({len(num_channels)}), downsample_parameters ({len(downsample_parameters)}) "
            f"and upsample_parameters ({len(upsample_parameters)}) must match."
        )
        assert all(len(x) == 4 for x in downsample_parameters), (
            f"Each element in downsample_parameters must have length 4, but has length {len(x)}."
        )
        assert all(len(x) == 5 for x in upsample_parameters), (
            f"Each element in upsample_parameters must have length 5, but has length {len(x)}."
        )

        # set up the VQVAE model
        self.model_class = "VQVAE"
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_channels = (num_res_channels for _ in range(len(num_channels)))
        self.num_res_layers = num_res_layers
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_init = embedding_init
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.dropout = dropout
        self.act = act
        self.output_act = output_act
        self.ddp_sync = ddp_sync
        self.use_checkpointing = use_checkpointing
        self.lr = learning_rate
        self.suffix = suffix

        self.net = VQVAE(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_channels=self.num_channels,
            num_res_channels=self.num_res_channels,
            num_res_layers=self.num_res_layers,
            downsample_parameters=self.downsample_parameters,
            upsample_parameters=self.upsample_parameters,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            embedding_init=self.embedding_init,
            commitment_cost=self.commitment_cost,
            decay=self.decay,
            epsilon=self.epsilon,
            dropout=self.dropout,
            act=self.act,
            output_act=self.output_act,
            ddp_sync=self.ddp_sync,
            use_checkpointing=self.use_checkpointing,
            )

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, ids = batch

        # forward pass
        imgs_recon, loss_quant = self.net(imgs)
        loss_recon = F.mse_loss(imgs_recon, imgs)
        loss = loss_recon + loss_quant

        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": loss, "imgs_recon": imgs_recon.detach(), "imgs": imgs, "ids": ids}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, ids = batch

        imgs_recon, loss_quant = self.net(imgs)
        loss_recon = F.mse_loss(imgs_recon, imgs)
        loss = loss_recon + loss_quant

        self.log("loss_val", loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, ids = batch

        imgs_recon, loss_quant = self.net(imgs)
        loss_recon = F.mse_loss(imgs_recon, imgs)
        loss = loss_recon + loss_quant

        # save loss
        self.log(
            "loss_test",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # save inferred masks
        for img, i in zip(imgs_recon, ids):
            if "out" in self.trainer.datamodule.data_test.data.columns:
                path_img = self.trainer.datamodule.data_test.data.out[
                    i.item()
                ]
            else:
                path_img = os.path.basename(self.trainer.datamodule.data_test.data.image[i.item()])
                path_img = path_img.split('.')
                path_img[0] = path_img[0] + self.suffix
                path_img = ".".join(path_img)
                path_img = os.path.join(
                    self.dir_out, path_img
                )
            # TODO: check if I need this or if I just use io.imsave
            save_image_mod(img, path_img, nrow=1, padding=0)

        # TODO: add step where we save embeddings using https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/vqvae.py#L445

        return loss_test

    def predict_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, ids = batch

        imgs_recon, _ = self.net(imgs)

        # save inferred masks
        for img, i in zip(imgs_recon, ids):
            if "out" in self.trainer.datamodule.data_predict.data.columns:
                path_img = self.trainer.datamodule.data_predict.data.out[
                    i.item()
                ]
            else:
                path_img = os.path.basename(self.trainer.datamodule.data_predict.data.image[i.item()])
                path_img = path_img.split('.')
                path_img[0] = path_img[0] + self.suffix
                path_img = ".".join(path_img)
                path_img = os.path.join(
                    self.dir_out, path_img
                )
            # TODO: check if I need this or if I just use io.imsave
            save_image_mod(img, path_img, nrow=1, padding=0)

        # TODO: add step where we save embeddings using https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/vqvae.py#L445

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_test_start(self):
        self.dir_out = os.path.join(self.trainer.logger.log_dir, "reconstructions")
        os.makedirs(
            self.dir_out,
            exist_ok=True,
        )

    def on_predict_start(self):
        self.dir_out = os.path.join(self.trainer.logger.log_dir, "reconstructions")
        os.makedirs(
            self.dir_out,
            exist_ok=True,
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        # save input variables which are not in the __init__function on checkpoints
        print("\non_save_checkpoint hook works!!!!\n")
        pass

    def on_load_checkpoint(self, checkpoint) -> None:
        # load input variables which are not in the __init__function on checkpoints
        print("\non_load_checkpoint hook works!!!!\n")
        pass
