########################################################################################################################
# This script splits a feature DataFrame into single sample numpy arrays.                                              #
# Author:               Daniel Schirmacher                                                                             #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                   #
# Python:               3.12.7                                                                                         #
########################################################################################################################
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def arg_parse():
    """
    Catch user input.


    Parameter
    ---------

    -


    Return
    ------

    Returns a namespace from `argparse.parse_args()`.
    """
    desc = (
        "Program to split a feature dataframe into single sample numpy arrays."
    )
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--features",
        type=str,
        default=r"N:\schroeder\Data\DS\PhD\nucleus_classification\data\3d\images\segmentation\mouse\cd41\rep4\subset\7_classic_features\classic_features_3D_c0_0-68_1000-2400_3100-4500.csv",
        help="Path to CSV file containing regionprops_table dataframe.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=r"N:\schroeder\Data\DS\PhD\nucleus_classification\data\3d\images\segmentation\mouse\cd41\rep4\subset\7_classic_features\split_1000",
        help="Path to output directory.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="cd41_val1000",
        help="Prefix for output file names. Will be used as '{prefix}_{mask_id}_cfeats.npy'.",
    )

    return parser.parse_args()


def main():
    args = arg_parse()

    path_f = args.features
    prefix = args.prefix
    path_out = args.out

    # load data
    df = pd.read_csv(path_f)

    # create output directory
    os.makedirs(path_out, exist_ok=True)

    # loop over each mask and save npy file
    mask_ids = df.pop("mask_id")

    with tqdm(total=len(mask_ids)) as pbar:
        for i, mask_id in enumerate(mask_ids):
            row = df.iloc[i].to_numpy().reshape(1, -1)
            path_out_i = os.path.join(
                path_out, f"{prefix}_{mask_id}_cfeats.npy"
            )
            np.save(path_out_i, row)

            pbar.update(1)


if __name__ == "__main__":
    main()