#######################################################################################################################
# This script contains custom lightning callbacks.                                                                    #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.12.7                                                                                        #
# PyTorch Version:      2.4.1                                                                                         #
# PyTorch Lightning Version: 2.4.0                                                                                    #
#######################################################################################################################
import torch
from lightning.pytorch.callbacks import Callback


class CheckpointCallback(Callback):
    """
    If checkpoint is loaded run validation once to update best loss for model saving.
    """

    def __init__(self, retrain: bool = False):
        super().__init__()

        self.retrain = retrain

    def on_fit_start(self, trainer, pl_module):
        # get callback ids of loss_val
        if self.retrain:
            cb_ids = [
                hasattr(cb, "monitor")
                and cb.monitor in ("loss_val")
                and cb.best_model_score is not None
                for cb in trainer.callbacks
            ]
            cb_ids = [i for i, val in enumerate(cb_ids) if val]

            # update best_model_score if we are retraining
            if any(cb_ids):
                for i in cb_ids:
                    # we assume the user has tested pretrained model and it is not sufficient
                    # --> we do not require a baseline for the untrained model
                    trainer.callbacks[i].best_model_score = torch.tensor(100.0)
                    trainer.callbacks[i].best_k_models[
                        trainer.callbacks[i].best_model_path
                    ] = torch.tensor(100.0)
