########################################################################################################################
# This script creates a distance matrix for 3D mask coordinates                                                        #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 26.03.2024                                                                                     #                                                                                          #
########################################################################################################################
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import argparse


def args_parse():
    desc = 'Script that computes a distance matrix for all pairs of masks.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--coordinates_path',
        type=str,
        required=True,
        help='Path to the mask coordinates .pkl file.'
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help=(
            'Path to where the output matrix .npy file should be saved '
            'including filename, excluding file extenion.'
        )
    )

    parser.add_argument(
        '--distance_function',
        type=str,
        default='euclidean',
        help=(
            'Metric with which the distance should be computet. Must be '
            'recognized by scipy.spatial.distance.cdist. Default: euclidean.'
        )
    )

    parser.add_argument(
        '--sr_z',
        type=float,
        default=1.,
        help='Sampling rate z-axis in um.'
    )

    parser.add_argument(
        '--sr_y',
        type=float,
        default=.24,
        help='Sampling rate y-axis in um.'
    )

    parser.add_argument(
        '--sr_x',
        type=float,
        default=.24,
        help='Sampling rate x-axis in um.'
    )

    return parser.parse_args()


def main():
    args = args_parse()
    coordinates_path = args.coordinates_path
    save_path = args.save_path
    distance_function = args.distance_function
    sr_z = args.sr_z
    sr_y = args.sr_y
    sr_x = args.sr_x

    with open(coordinates_path, 'rb') as f:
        coordinates = pickle.load(f)

    centroids = np.array([np.array([np.mean(z) * sr_z, np.mean(y) * sr_y, np.mean(x) * sr_x])
                          for z, y, x in coordinates.values()])
    distance_matrix = cdist(centroids, centroids, str(distance_function))
    distance_matrix = np.array(distance_matrix, dtype=np.float16)
    np.save(save_path, distance_matrix)
    print(f'Matrix with {distance_function} metric saved successfully.')


if __name__ == '__main__':
    main()
