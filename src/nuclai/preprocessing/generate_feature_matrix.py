########################################################################################################################
# This script creates a matrix of 3D features for every single mask.                                                   #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 07.03.2024                                                                                     #                                                                                          #
########################################################################################################################
import numpy as np
import argparse
import glob
import os
import imageio.v2 as imageio
from skimage import measure
import pickle
import difflib
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from histomicstk.features import compute_nuclei_features
import pandas as pd
import ipdb

def args_parse():
    '''
     Catches user input from the CLI.

    Parameters

    ----------
        
    -

    Return

    ------

    Returns a namespace from `argparse.parse_args()`.

    '''
    desc = 'Program to generate feature matrices for segmented nuclei.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--image_dir',
        type=str,
        default=r'N:\schroeder\Data\LR\feature_test\img',
        help=(
        'Path to the directory that contains all nuclear images. '
        'Expects shape ZYX.'
        )
    )   

    parser.add_argument(
        '--mask_dir',
        type=str,
        default=r'N:\schroeder\Data\LR\feature_test\mask',
        help=(
        'Path to the directory that contains the corresponding '
        'nuclear masks to the images. Image and mask filename pairs should '
        'share the same beginning. Expects shape ZYX.' 
        )
    )

    parser.add_argument(
        '--filetype',
        type=str,
        default='.tif',
        help=(
        'Specify which filetype the image files and mask files have. '
        'E.g. \'.tif\', which are the preferred filetype'
        )
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default=r'N:\schroeder\Data\LR\feature_test\output',
        help='Filepath where to save the output .csv file under.'
    )

    return parser.parse_args()


def generate_features(img: np.ndarray, mask: dict) -> np.ndarray:
    """
    Wrapper that calls the individual feature generating functions and
    concatenates their output to a single feature vector for each individual 
    segmentation mask. 

    Parameter
    ------

    img: np.ndarray
        Image of shape ZYX. Expected to match shape of mask.

    mask: np.ndarray
        Mask array of shape ZYX. Expected to match shape of img.

    Returns:
    ------
    Feature matrix of shape (instances, features). Feature names is a list of 
    length number of features. It holds the names of all generated features.
    
    """
    highest_idx = max(np.unique(mask))
    feature_matrix = np.full((highest_idx, 104), np.nan)

    # Generates the HistomicsTK features for all masks in 3D
    histomics_3d = np.full((highest_idx, 88, img.shape[0]), np.nan)
    histomics_names = compute_nuclei_features(mask[1, :, :], img[1, :, :]).columns
    histomics_names = histomics_names.tolist()
    for z in tqdm(range(0, img.shape[0]), ncols=100, desc='\n### Computing HistomicsTK features'):
        if len(np.unique(mask[z, :, :])) > 1:
            histomics_z = compute_nuclei_features(mask[z, :, :], img[z, :, :]).values
            for row in range(0, histomics_z.shape[0]):
                histomics_3d[int(histomics_z[row, 0] - 1), :, z] = histomics_z[row, :]
        else:
            print(f'\n### No masks found in z-layer #{z}.')
    histomics_avg = np.nanmean(histomics_3d, axis=2)

    # Alternative Regionprops feature in 3D
    regionprops_3d = np.full((highest_idx, 16, img.shape[0]), np.nan)
    for z in tqdm(range(0, img.shape[0]), ncols=100, desc='\n### Computing sklearn regionprops features'):
        if len(np.unique(mask[z, :, :])) > 1:
            regionprops_z = measure.regionprops(mask, img, cache=True)
            for i in range(0, len(regionprops_z)):
                label = regionprops_z[i].label
                regionprops_3d[(label-1), 0, z] = regionprops_z[i].axis_major_length
                regionprops_3d[(label-1), 1, z] = regionprops_z[i].euler_number
                inertia_tensor = regionprops_z[i].inertia_tensor.ravel()
                regionprops_3d[(label-1), 2:11, z] = inertia_tensor
                inertia_tensor_eigvals = regionprops_z[i].inertia_tensor_eigvals[:]
                regionprops_3d[(label-1), 11:14, z] = inertia_tensor_eigvals
        else:
            print(f'\n### No masks found in z-layer index {z}.')
    regionprops_avg = np.nanmean(regionprops_3d, axis=2)
    feature_matrix[:, 0:88] = histomics_avg
    feature_matrix[:, 88:] = regionprops_avg
    feature_matrix = feature_matrix[:, :-2] #drop the last two columns, as they somehow always turn out NaN
    not_all_nan_rows = ~np.all(np.isnan(feature_matrix), axis=1)
    feature_matrix = feature_matrix[not_all_nan_rows]

    return feature_matrix, histomics_names


def main():
    args = args_parse()
    image_directory = args.image_dir
    mask_directory = args.mask_dir
    filetype = args.filetype
    out_dir = args.out_dir
    
    os.makedirs(out_dir, exist_ok=True)


    # Detects all filenames in the user-specified input directories
    img_filenames = glob.glob(os.path.join(image_directory, f'*{filetype}'))
    mask_filenames = glob.glob(os.path.join(mask_directory, f'*{filetype}'))

    img_filenames.sort()
    mask_filenames.sort()
    print(f'\n### Detected {len(img_filenames)} images, {len(mask_filenames)} mask-files.')

    assert len(img_filenames) == len(mask_filenames), f'Error: Detected {len(img_filenames)} images and {len(mask_filenames)}.'

    # Loads the pairs of img-file and corresponding mask-file
    for img_name, mask_name in zip(img_filenames, mask_filenames):
        
        # Loads img to nparray
        img_path = os.path.join(image_directory, img_name)
        img = imageio.imread(img_path)
        img = np.array(img)
        print(f'\n### Image of shape {img.shape} loaded.')
    

        # Loads masks to npndarray
        mask_path = os.path.join(mask_directory, mask_name)
        mask = imageio.imread(mask_path)
        mask = np.array(mask, dtype=np.uint32)
        print(f'\n### Mask array of shape {mask.shape} loaded.')

        assert img.shape == mask.shape, f'Error image with name {img_name} and mask with name {mask_name} are of different shape:{img.shape} and {mask.shape}'

        feature_matrix, histomics_names = generate_features(img, mask) 
        
        csv_name = str(os.path.splitext(os.path.basename(img_name))[0]) + '_feature_matrix.csv'
        final_outdir = os.path.join(out_dir, csv_name)
        np.savetxt(final_outdir, feature_matrix, delimiter=',')
        print(f'\n### Created .csv-file for feature-matrix of image {img_name}.')
    
    # save feature names
    sklearn_feature_names = [
        'axis_major_length',
        'euler_number',
        'inertia_tensor_1',
        'inertia_tensor_2',
        'inertia_tensor_3',
        'inertia tensor_4',
        'inertia_tensor_5',
        'inertia_tensor_6',
        'inertia_tensor_7',
        'inertia_tensor_8',
        'inertia_tensor_9',
        'inertia_tensor_eigvals_1',
        'inertia_tensor_eigvals_2',
        'inertia_tensor_eigvals_3'
        ]
    
    feature_names = histomics_names + sklearn_feature_names
    feature_names = np.array(feature_names)
    np.savetxt(os.path.join(out_dir, 'feature_names.csv'), feature_names, fmt='%s', delimiter=',')
    print(f'\n### Feature names vector saved to {out_dir}.')
        
if __name__ == '__main__':
    main()
