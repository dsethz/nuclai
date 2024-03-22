########################################################################################################################
# This script generates the dataset structure used as input for GNNs.                                                  #
# Author:               Daniel Schirmacher                                                                             #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                   #
# Date:                 08.03.2024                                                                                     #
# Python:               3.11.7                                                                                         #
########################################################################################################################
import argparse
import glob
import os
import torch
import ipdb

import pandas as pd
import torch_geometric as tg


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
    desc = "Program to obtain a .csv file containing matching bf/mask paths in each row."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--adjacency",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/images/*.png') to bright field images. Naming convention must match "
        "naming convention of --mask s.t. alphanumerically sorted paths are matching.",
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/masks/*.png') to segmentation masks.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/masks/*.png') to segmentation masks.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="train",
        help="Prefix for output file name (i.e. '{prefix}_paths.csv'). Default is 'train'.",
    )

    return parser.parse_args()


def _generate_df(path_bf: str, path_mask: str) -> pd.DataFrame:
    files_bf = glob.glob(path_bf)
    files_bf.sort()
    files_mask = glob.glob(path_mask)
    files_mask.sort()

    df = pd.DataFrame({"bf": files_bf, "mask": files_mask})

    return df


def main():
    args = arg_parse()

    path_adjacency = args.adjacency
    path_features = args.features
    path_labels = args.labels
    path_out = args.out

    # convert adjacency matrix to sparse pyg tensor with tg.utils.dense_to_sparse
    adjacency = np.load(path_adjacency)
    adjacency = torch.from_numpy(adjacency)
    adjacency = tg.utils.dense_to_sparse(adjacency)

    ipdb.set_trace()

    df = _generate_df(path_bf, path_mask)
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(os.path.join(path_out, f"{prefix}_paths.csv"), index=None)


if __name__ == "__main__":
    main()
