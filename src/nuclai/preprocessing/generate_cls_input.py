########################################################################################################################
# This script constructs a .json file containing paths to features and the corresponding label.                        #
# Author:               Daniel Schirmacher                                                                             #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                   #
# Python:               3.12.7                                                                                         #
########################################################################################################################
import argparse
import glob
import json
import os


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
        "Program to obtain a .json file containing feature paths and labels."
    )
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/features/*.npy') to features.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path label .json file. Assumes that labels are stored under key 'ctype'.",
    )

    parser.add_argument(
        "--class_pos",
        type=str,
        default="positive",
        help="Name of positive class used in labels .json file. Assumes that labels are stored under key 'ctype'.",
    )

    parser.add_argument(
        "--class_neg",
        type=str,
        default="negative",
        help="Name of negative class used in labels .json file.  Assumes that labels are stored under key 'ctype'.",
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
        default="cd41",
        help="Prefix for output file name (i.e. '{prefix}_data.json'). Default is 'cd41'.",
    )

    return parser.parse_args()


def main():
    args = arg_parse()

    path_f = args.features
    path_l = args.labels
    class_pos = args.class_pos
    class_neg = args.class_neg
    prefix = args.prefix
    path_out = args.out

    # create output directory
    os.makedirs(path_out, exist_ok=True)

    # get feature paths
    features = glob.glob(path_f)
    features.sort()

    # load data
    with open(path_l) as f:
        labels = json.load(f)

    # create dictionary
    data = {}
    i, j = 0, 0
    while i < len(features):
        path = features[i]
        idx = os.path.basename(path).split(".")[0].split("_")[-2]

        if labels[idx]["ctype"] == class_pos:
            tmp = {"path": path, "label": 1}
            data[j] = tmp
            i += 1
            j += 1
        elif labels[idx]["ctype"] == class_neg:
            tmp = {"path": path, "label": 0}
            data[j] = tmp
            i += 1
            j += 1
        else:
            i += 1

    print(f"Number of kept samples: {j}")
    print(f"Number of removed samples: {i - j}")

    with open(os.path.join(path_out, f"{prefix}_data.json"), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
