########################################################################################################################
# This is a pytorch lightning wrapper around MONAI's VQ-VAE.                                                           #
# Author:               Daniel Schirmacher                                                                             #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.12.7                                                                                         #
# PyTorch Version:      2.4.1                                                                                          #
# Lightning Version:    2.4.0                                                                                          #
########################################################################################################################
# Imports
import os
from typing import Optional

import lightning as L
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score


class MLP(nn.Module):
    """
    MLP in torch.

    Args:
        input_dim: int
            #features of the input.
        output_dim: int
            #features of the output (i.e. #classes).
        hidden_dim: int
            #neurons per hidden MLP layer. Each element of list corresponds to the number
            of neurons of its respective layer (e.g. first element defines #neurons in
            the layer receiving input from the feature_extractor, etc.). If hidden_dim
            is an empty list or None, no hidden layers will be used.
        bias: bool
            Whether to use bias in the linear layers.
        dropout: float
            Dropout rate. If 0.0, no dropout is used. Must be in [0, 1].

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[list[int]] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # assert input formats
        hidden_dim = hidden_dim if hidden_dim is not None else []
        assert isinstance(
            input_dim, int
        ), f'input_dim is expected to be of type "int", but is of type "{type(input_dim)}".'
        assert isinstance(
            output_dim, int
        ), f'output_dim is expected to be of type "int", but is of type "{type(output_dim)}".'
        assert hidden_dim is None or isinstance(
            hidden_dim, list
        ), f'hidden_dim is expected to be "None" or of type list, but is of type"{type(hidden_dim)}".'
        assert all(
            isinstance(v, int) for v in hidden_dim
        ), 'elements of hidden_dim are expected to be of type "int".'
        assert isinstance(
            bias, bool
        ), f'bias is expected to be of type "bool", but is of type "{type(bias)}".'
        assert isinstance(
            dropout, float
        ), f'dropout is expected to be of type "float", but is of type "{type(dropout)}".'
        assert 0.0 <= dropout <= 1.0, "dropout must be in [0, 1]."

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dropout = dropout
        self.net = []

        # set up classifier
        if self.hidden_dim is not None:
            dimensions = [input_dim] + self.hidden_dim + [self.output_dim]
        else:
            dimensions = [input_dim, self.output_dim]

        for i in range(len(dimensions) - 1):
            dim_in = dimensions[i]
            dim_out = dimensions[i + 1]
            self.net.append(nn.Linear(dim_in, dim_out, bias=self.bias))

            # remove sigmoid due to numerical instability with mixed-precision
            # if i == (len(dimensions) - 2):
            #     self.net.append(nn.Sigmoid())
            self.net.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
            self.net.append(nn.BatchNorm1d(dimensions[i + 1]))
            if self.dropout > 0.0:
                self.net.append(nn.Dropout(p=self.dropout))

        # add the list of modules to the current module
        self.net = nn.ModuleList(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class LitMLP(L.LightningModule):
    """
    Set up of the LightningModule for MONAI's VQVAE.

    Args:
        input_dim (int): Input dimensionality.
        output_dim (int): Output dimensionality (i.e. #classes).
        hidden_dim (list[int]): Number of neurons per hidden layer.
        bias (bool): Whether to use bias in the linear layers.
        dropout (float): Dropout rate. Must be in [0, 1].
        loss_weight (float): Weight of the loss.
        learning_rate (float): Learning rate.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: list[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        loss_weight: float = 1.0,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        # assert input formats
        assert isinstance(
            input_dim, int
        ), f'input_dim is expected to be of type "int" but is of type "{type(input_dim)}".'
        assert hidden_dim is None or isinstance(
            hidden_dim, list
        ), f'hidden_dim is expected to be of type "list" but is of type "{type(hidden_dim)}".'
        assert isinstance(
            bias, bool
        ), f'bias is expected to be of type "bool" but is of type "{type(bias)}".'
        assert isinstance(
            dropout, float
        ), f'dropout is expected to be of type "float" but is of type "{type(dropout)}".'
        assert isinstance(
            loss_weight, float
        ), f'loss_weight is expected to be of type "float" but is of type "{type(loss_weight)}".'
        assert isinstance(
            learning_rate, float
        ), f'learning_rate is expected to be of type "float" but is of type "{type(learning_rate)}".'

        # additional assertions
        if hidden_dim is not None:
            assert all(
                isinstance(v, int) for v in hidden_dim
            ), 'elements of hidden_dim are expected to be of type "int".'

        assert 0.0 <= dropout <= 1.0, "dropout must be in [0, 1]."

        # save hyperparameters from __init__ upon checkpoints in hparams.yaml
        self.save_hyperparameters()

        # set up the VQVAE model
        self.model_class = "MLP"
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dropout = dropout
        self.loss_weight = torch.FloatTensor([loss_weight])
        self.lr = learning_rate

        # set up the model
        self.net = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            bias=self.bias,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, labels, ids = batch

        # forward pass
        prediction = self.net(features)
        loss = F.binary_cross_entropy_with_logits(
            prediction, labels, weight=self.loss_weight
        )

        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return {
            "loss": loss,
            "prediction": prediction.detach(),
            "labels": labels,
            "ids": ids,
        }

    def on_validation_start(self):
        self.f1_val = F1Score(task="binary", threshold=0.5)

        # get sigmoid for predictions
        self.sigmoid = nn.Sigmoid()

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, labels, ids = batch

        prediction = self.net(features)

        # log loss
        loss = F.binary_cross_entropy_with_logits(
            prediction, labels, weight=self.loss_weight
        )

        self.log("loss_val", loss, on_step=True, on_epoch=True, sync_dist=True)

        # convert prediction to probability for F1
        prediction = self.sigmoid(prediction)

        # update f1_val
        self.f1_val.update(
            torch.squeeze(prediction, -1), torch.squeeze(labels, -1)
        )

        return loss

    def on_validation_epoch_end(self):
        f1 = self.f1_val.compute()
        self.log("f1_val", f1, sync_dist=True)

    def on_test_start(self):
        # create prediction data frame to which all predictions are appended
        self.prediction_data = pd.DataFrame(
            columns=["id", "prediction", "label"]
        )

        # set up f1 metric
        self.f1_test = F1Score(task="binary", threshold=0.5)

        # get sigmoid for predictions
        self.sigmoid = nn.Sigmoid()

    def test_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, labels, ids = batch

        prediction = self.net(features)
        loss = F.binary_cross_entropy_with_logits(
            prediction, labels, weight=self.loss_weight
        )

        # save loss
        self.log(
            "loss_test",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        # convert prediction to probability
        prediction = self.sigmoid(prediction)

        # update f1_test
        self.f1_test.update(
            torch.squeeze(prediction, -1), torch.squeeze(labels, -1)
        )

        # save predictions and labels
        idx = os.path.basename(
            self.trainer.datamodule.data_test.data[str(ids[0].item())]["path"]
        )

        tmp = pd.DataFrame(
            {
                "id": [idx],
                "prediction": [prediction.cpu().detach().numpy().item()],
                "label": [labels.cpu().detach().numpy().item()],
            }
        )
        self.prediction_data = pd.concat([self.prediction_data, tmp])

        return loss

    def on_test_epoch_end(self):
        # save prediction data frame
        self.prediction_data.to_csv(
            os.path.join(self.trainer.logger.log_dir, "predictions.csv"),
            index=False,
        )

        # compute f1 score
        f1 = self.f1_test.compute()
        self.log("f1_test", f1, sync_dist=True)

    def on_predict_start(self):
        # create prediction data frame to which all predictions are appended
        self.prediction_data = pd.DataFrame(columns=["id", "prediction"])

        # get sigmoid for predictions
        self.sigmoid = nn.Sigmoid()

    def predict_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, ids = batch

        prediction = self.net(features)

        # convert prediction to probability
        prediction = self.sigmoid(prediction)

        # save predictions and labels
        idx = os.path.basename(
            self.trainer.datamodule.data_predict.data[str(ids[0].item())][
                "path"
            ]
        )
        tmp = pd.DataFrame(
            {
                "id": [idx],
                "prediction": [prediction.cpu().detach().numpy().item()],
            }
        )
        self.prediction_data = pd.concat([self.prediction_data, tmp])

    def on_predict_end(self):
        # save prediction data frame
        self.prediction_data.to_csv(
            os.path.join(self.trainer.logger.log_dir, "predictions.csv"),
            index=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-8,
            ),
            "monitor": "loss_val",
            "frequency": 1,
            "name": "reduce_lr_on_plateau",
        }
        return [optimizer], [lr_scheduler]

    def on_save_checkpoint(self, checkpoint) -> None:
        # save input variables which are not in the __init__function on checkpoints
        pass

    def on_load_checkpoint(self, checkpoint) -> None:
        # load input variables which are not in the __init__function on checkpoints
        pass
