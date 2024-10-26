########################################################################################################################
# This script contains evaluation metrics we use during model training and testing.                                    #
# Author:               Daniel Schirmacher                                                                             #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.12.7                                                                                         #
# PyTorch Version:      2.4.1                                                                                          #
# Lightning Version:    2.4.0                                                                                          #
########################################################################################################################
# Imports
import torch
import torch.nn.functional as F


def mse_loss_masked(
    output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the mean squared error loss between the output and target tensors, ignoring masked values.

    Args:
        output: torch.Tensor
            The output tensor.
        target: torch.Tensor
            The target tensor.
        mask: torch.Tensor
            The mask tensor.

    Returns:
        torch.Tensor
            The mean squared error loss.
    """
    loss = F.mse_loss(output, target, reduction="none")
    loss = loss * mask
    loss = loss.sum() / mask.sum()

    return loss
