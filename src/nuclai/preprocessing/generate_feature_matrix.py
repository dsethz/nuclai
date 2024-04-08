########################################################################################################################
# This script creates a matrix of 3D features for every single mask.                                                   #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 07.03.2024                                                                                     #
########################################################################################################################
import argparse
import difflib
import glob
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from histomicstk.features import compute_nuclei_features
from skimage import measure
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


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
        '--mask_dict_dir',
        type=str,
        default=r'N:\schroeder\Data\LR\feature_test\mask_dict',
        help=('Path to .pkl file of mask coordinates. Saves computation '
        'time for big images.'
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


def detect_common_name(img_name: str, mask_name: str, coord_name: str) -> str:
    """
    Given the three filenames, finds the common substring in the beginning.

    Parameter
    ------

    img_name: str
        Name of the image file. 

    mask_name: str
        Name of the mask file. 

    coord_name: str
        Name of the masks-coordinates pickle file. 

    Returns:
    ------
    Common substring at the beginning of the three filenames.
    
    """
    # Use difflib to find the longest common substring
    sequences = [img_name, mask_name, coord_name]
    common_substring = ""
    matcher = difflib.SequenceMatcher(None, *sequences)
    match = matcher.find_longest_match(0, len(img_name), 0, len(mask_name))
    if match.size > len(common_substring):
        common_substring = img_name[match.a:match.a + match.size]
    return common_substring


def generate_mask_dict(masks: np.ndarray) -> tuple:
    """
    Transforms the masks-array into a computationally more efficient dictionary.

    Parameter
    ------

    masks: np.ndarray
        The masks array of shape z, y, x and same shape as image array. 

    Returns:
    ------

    Returns a dict with keys being mask indices and values being a tuple of
    three np.ndarrays storing the coordinate components of the mask as such
    (array(z), array(y), array(x)). 
    """
    coordinates = {}
    # Iterating through the 3D mask
    for i in tqdm(range(masks.shape[0]), position=0, leave=True, desc="\n### Convert mask array to dictionary", ncols=100):
        for j in range(masks.shape[1]):
            for k in range(masks.shape[2]):
                value = masks[i, j, k]
                if value not in coordinates:
                    coordinates[value] = ([], [], [])
                coordinates[value][0].append(i)
                coordinates[value][1].append(j)
                coordinates[value][2].append(k)
    del coordinates[0] #key 0 indicates background, we only want to save the masks

    # Convert lists to tuples of arrays
    for value in tqdm(coordinates, desc="\n### Wrapping mask dictionary", ncols=100):
        coordinates[value] = tuple(np.array(coord_list) for coord_list in coordinates[value])
    return coordinates


def generate_features(img: np.ndarray, mask: dict, coordinates: np.ndarray) -> np.ndarray:
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
    feature_matrix = np.full((max(coordinates.keys()), 104), np.nan)

    # Generates the HistomicsTK features for all masks in 3D
    histomics_3d = np.full((max(coordinates.keys()), 88, img.shape[0]), np.nan)
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
    regionprops_3d = np.full((max(coordinates.keys()), 16, img.shape[0]), np.nan)
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
    mask_coord_directory = args.mask_dict_dir
    filetype = args.filetype
    out_dir = args.out_dir
 
    os.makedirs(out_dir, exist_ok=True)
    
    # Detects all filenames in the user-specified input directories
    img_filenames = glob.glob(os.path.join(image_directory, f'*{filetype}'))
    mask_filenames = glob.glob(os.path.join(mask_directory, f'*{filetype}'))
    mask_coord_filenames = glob.glob(os.path.join(mask_coord_directory, '*.pkl'))

    img_filenames.sort()
    mask_filenames.sort()
    mask_coord_filenames.sort()

    assert len(img_filenames) == len(mask_filenames) == len(mask_coord_filenames), f'Error: Detected {len(img_filenames)} images, {len(mask_filenames)} mask-files and {len(mask_coord_filenames)} mask-coordinate.pkl files.'

    # Creates mask-coordinate .pkl files if missing and subsequently detects their names in the directory
    if len(mask_filenames) > len(mask_coord_filenames):
        print('\n### Could not find a corresponding .pkl file of mask coordinates for every mask file.')
        print('\n### Generating .pkl file of mask coordinates from mask file for every mask file in the directory.')
        for mask_name in mask_filenames:
            masks = imageio.imread(os.path.join(mask_directory, mask_name))
            mask_coord_filenames = load_filenames(image_directory, mask_directory, mask_coord_directory, filetype)[2]
            mask_dict = generate_mask_dict(masks)
            mask_coord_name = 'coordinates_' + str(mask_name[:-4]) + '.pkl'
            save_dir = os.path.join(mask_coord_directory, mask_coord_name)
            with open(rf'{save_dir}', 'wb') as f:
                pickle.dump(mask_dict, f)
            print(f'\n### Saved pickle file of mask coordinates for {mask_name[:-4]} in {mask_coord_directory}.')
            print(f'\n### Detected {len(img_filenames)} images, {len(mask_filenames)} mask-files and {len(mask_coord_filenames)} mask-coordinate.pkl files.')


    # Loads the pairs of img-file and corresponding mask-file
    for img_name, mask_name, coord_name in zip(img_filenames, mask_filenames, mask_coord_filenames):
        
        # Loads img to nprray
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

        # Loads .pkl file of mask coordinate dictionary
        path = os.path.join(mask_coord_directory, coord_name)
        with open(path, 'rb') as f:
            coordinates = pickle.load(f)
        print(f'\n### Coordinates of {len(coordinates.keys())} masks loaded.')

        feature_matrix, histomics_feature_names = generate_features(img, mask, coordinates) 
        
        #feature_matrix.dropna(how='all')
        #feature_name_df = pd.DataFrame(feature_names)
        feature_names_path = os.path.join(out_dir, str('feature_names.csv'))
        #feature_name_df.to_csv(feature_names_path, index=False, header=False)
        csv_name = str(detect_common_name(img_name, mask_name, coord_name)) + '_feature_matrix_big.csv'
        final_outdir = os.path.join(out_dir, csv_name)
        np.savetxt(final_outdir, feature_matrix, delimiter=',')
        print(f'\n### Created .csv-file for feature-matrix of image {img_name}.')
    
    # save feature names
    feature_names = [
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

    feature_names = histomics_feature_names + feature_names
    feature_names = np.array(feature_names)
    np.savetxt(os.path.join(out_dir, 'feature_names.csv'), feature_names, fmt='%s', delimiter=',')
        
        
if __name__ == '__main__':
    main()
