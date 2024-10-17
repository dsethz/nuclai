########################################################################################################################
# This script constructs a .csv file containing paths to images.                                                       #
# Author:               Daniel Schirmacher                                                                             #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                   #
# Date:                 17.10.2024                                                                                     #
# Python:               3.12.7                                                                                         #
########################################################################################################################
import argparse
import glob
import os

import pandas as pd


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
    desc = "Program to obtain a .csv file containing image paths in each row."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/images/*.png') to images.",
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


def _generate_df(path_img: str) -> pd.DataFrame:
    files_img = glob.glob(path_img)
    files_img.sort()

    df = pd.DataFrame({"image": files_img})

    return df


def main():
    args = arg_parse()

    path_img = args.img
    prefix = args.prefix
    path_out = args.out

    df = _generate_df(path_img)
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(os.path.join(path_out, f"{prefix}_paths.csv"), index=None)


if __name__ == "__main__":
    main()
