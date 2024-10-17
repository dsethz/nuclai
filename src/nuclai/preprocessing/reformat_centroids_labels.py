########################################################################################################################
# This script reformats a single json file containing labels and centroids.                                            #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# Date:                 12.09.2024                                                                                     #
########################################################################################################################
import argparse
import ipdb
import json
import os

import numpy as np
import pandas as pd


def args_parse():
    desc = 'Script that reformats a single json file containing labels and centroids.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--centroids',
        type=str,
        required=True,
        help='Path to centroids and labels .json file.'
    )

    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to features .csv file.'
    )

    parser.add_argument(
        '--out_centroids',
        type=str,
        required=True,
        help='Directory to which centroids .csv file will be saved.'
    )

    parser.add_argument(
        '--out_labels',
        type=str,
        required=True,
        help='Directory to which labels .csv file will be saved.'
    )

    return parser.parse_args()


def main():
    args = args_parse()
    path_c = args.centroids
    path_f = args.features
    path_out_centroids = args.out_centroids
    path_out_labels = args.out_labels

    os.makedirs(path_out_labels, exist_ok=True)
    os.makedirs(path_out_centroids, exist_ok=True)

    features = pd.read_csv(path_f)
    with open(path_c, 'r') as f:
        centroids = json.load(f)

    # get ids that are in centroids but not in features
    mismatching_ids = np.setdiff1d(list(centroids.keys()), features.mask_id.tolist())

    # assign labels to df and centroids to new dict
    labels = pd.DataFrame(columns=['mask_id', 'label'])
    centroids_new = {}

    for k in centroids.keys():
        if not k in mismatching_ids:
            tmp = pd.DataFrame()
            

    ipdb.set_trace()


if __name__ == '__main__':
    main()
