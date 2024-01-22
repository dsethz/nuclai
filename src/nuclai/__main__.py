########################################################################################################################
# This script contains all entrypoints for training, testing, and predicting with cell nucleus classifiers.            #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# PyTorch Version:      2.1.2                                                                                          #
# Lightning Version:    2.1.3                                                                                          #
########################################################################################################################
# Imports
import argparse


def _args_inference(mode: str) -> argparse.Namespace:
    """
    This function sets up input arguments for train/test/predict.
    """
    # check mode
    assert mode in ["train", "test", "predict"], f"{mode} is an invalid mode."

    # get user input
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file containing test image file paths.",
    )

    return parser.parse_args()


def train():
    """
    This function coordinates model training.

    * maybe we want a cli wrapper(see aisegcell) for this?
    """
    test_string = "this is a very long string to test if ruff will auto line-break if we exceed 79 characters."
    print(test_string)


def test():
    """
    This function coordinates model testing.

    * maybe we want a cli wrapper(see aisegcell) for this?
    """


def predict():
    """
    This function coordinates model prediction.

    * maybe we want a cli wrapper(see aisegcell) for this?
    """
