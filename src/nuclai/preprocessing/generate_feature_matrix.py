########################################################################################################################
# This script generates a feature matrix from a list of input images.                                                  #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# PyTorch Version:      2.1.2                                                                                          #
# Lightning Version:    2.1.3                                                                                          #
########################################################################################################################
import numpy as np
import argparse
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
        '--image_directory',
        type = str,
        default = r'N:\schroeder\Data\LR\feature_test\img',
        help = 'Path to the directory that contains all nuclear images. Expects shape ZYX.'
    )   

    parser.add_argument(
        '--mask_directory',
        type = str,
        default = r'N:\schroeder\Data\LR\feature_test\mask',
        help = 'Path to the directory that contains the corresponding nuclear masks to the images. Image and mask filename pairs should share the same beginning. Expects shape ZYX.' 
    )

    parser.add_argument(
        '--mask_dict_path',
        type = str,
        default = r'N:\schroeder\Data\LR\feature_test\mask_dict',
        help = 'Path to .pkl file of mask coordinates. Saves computation time for big images.'
    )

    parser.add_argument(
        '--filetype',
        type = str,
        default = '.tif',
        help = 'Specify which filetype the image files and mask files have. E.g. \'.tif\', which are the preferred filetype'
    )

    parser.add_argument(
        '--out_dir',
        type = str,
        default = r'N:\schroeder\Data\LR\feature_test\output',
        help = 'Filepath where to save the output .csv file under.'
    )

    return parser.parse_args()



def get_avg_neigh_distance(coordinates: dict, n_nearest: int = 5) -> np.ndarray:
    """
    Calculates average distance of each mask centroid to its n-closest neighbour centroids.

    Parameter
    ------

    coordinates: dict
        Dictionary with keys holding mask indeces and values the corresponding tuple of (np.array[mask x-coords.], np.array[mask y-coords.], np.array[mask z-coords.]).
    
    n_nearest: int
        Optional integer defining the number n closest neighbours, which distance should be considered.

    Returns:
    ------

    Returns np.ndarray of shape (#masks, 2). First columns holds the mask index, second row holds the masks squared euclidian distance to its n closest neighbours centroids
    and a pairwise distance matrix between all mask centroids. 
    """
    centroid_matrix = np.zeros((len(coordinates.keys()), 4))
    avg_neigh_dist = np.zeros((len(coordinates.keys()), 2))

    # Calculates centroids.
    for mask_idx in coordinates.keys():
        centroid = round(int(np.average(coordinates[mask_idx][0]))), round(int(np.average(coordinates[mask_idx][1]))), round(int(np.average(coordinates[mask_idx][2])))
        centroid_matrix[(mask_idx-1), 0] = mask_idx
        centroid_matrix[(mask_idx-1), 1:] = centroid
    
    # Calculates pairwise distances and avg. distance of n-nearest centroids.
    dist_matrix = cdist(centroid_matrix, centroid_matrix, 'sqeuclidean')
    n_nearest_centroids = np.argsort(dist_matrix, axis=1)[:, 1:(n_nearest+1)]
    for mask_idx in coordinates.keys():
        avg_neigh_dist[(mask_idx-1), 0] = mask_idx
        avg_neigh_dist[(mask_idx-1), 1] = np.sum(dist_matrix[(mask_idx-1), n_nearest_centroids[(mask_idx-1), :]]) / n_nearest
    return avg_neigh_dist, dist_matrix


def load_filenames(image_directory: str, mask_directory: str, mask_coord_directory: str, filetype: str) -> tuple:
    """
    Finds all filenames in the input directories specified by the user.

    Parameter
    ------

    image_directory: str
        Filepath to the directory, where the image files can be found.

    mask_directory: str
        Filepath to the directory, where the mask files can be found.

    mask_directory: str
        Filepath to the directory, where the mask-coordinate.pkl files can be found.
    
    filetype: str
        Filetype of the image-files or mask files e.g. \'.tif\'.

    Returns:
    ------

    Returns tuple of three lists, holding the filenames found in image_directory, mask_directory and mask_coord_directory.
    """
    img_filenames = os.listdir(image_directory)
    mask_filenames = os.listdir(mask_directory)
    mask_coord_filenames= os.listdir(mask_coord_directory)

    img_filenames = [img for img in img_filenames if img.endswith(filetype)]
    mask_filenames = [mask for mask in mask_filenames if mask.endswith(filetype)]
    mask_coord_filenames = [coord for coord in mask_coord_filenames if coord.endswith('.pkl') ]

    assert (
        len(img_filenames) == len(mask_filenames)
        ), print(f'\n### ERROR: Expected number of image files and mask files to match. But found {len(img_filenames)} image files and {len(mask_filenames)} mask-files.')
    
    return img_filenames, mask_filenames, mask_coord_filenames


def generate_features(img: np.ndarray, mask: dict, coordinates: np.ndarray) -> np.ndarray:
    """
    Wrapper that calls the individual feature generating functions and concatenates their output to a single feature vector for each individual segmentation mask. 

    Parameter
    ------

    img: np.ndarray
        Image of shape ZYX. Expected to match shape of mask.

    mask: np.ndarray
        Mask array of shape ZYX. Expected to match shape of img.

    Returns:
    ------
    Feature vector of shape 
    
    """
    feature_matrix = np.full((len(coordinates.keys()), 104), np.nan)

    bboxes = bounding_box(coordinates)

    # Generates the HistomicsTK features for all masks in 3D
    histomics_3d = np.full((len(coordinates.keys()), 88, img.shape[0]), np.nan)
    for z in tqdm(range(0, img.shape[0]), ncols=100, desc='\n### Computing HistomicsTK features'):
        if len(np.unique(mask[z, :, :])) > 1:
            histomics_z = compute_nuclei_features(mask[z, :, :], img[z, :, :]).values
            for row in range(0, histomics_z.shape[0]):
                histomics_3d[int(histomics_z[row, 0] - 1), :, z] = histomics_z[row, :]
        else:
            print(f'\n### No masks found in z-layer #{z}.')
    histomics_avg = np.nanmean(histomics_3d, axis=2)
    

    # Alternative Regionprops feature in 3D
    regionprops_3d = np.full((len(coordinates.keys()), 16, img.shape[0]), np.nan)
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
            print(f'\n### No masks found in z-layer #{z}.')
    regionprops_avg = np.nanmean(regionprops_3d, axis=2)
    feature_matrix[:, 0:88] = histomics_avg
    feature_matrix[:, 88:] = regionprops_avg

    return feature_matrix


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

    Returns a dict with keys being mask indices and values being a tuple of three np.ndarrays storing the coordinate components of the mask as such (array(z), array(y), array(x)). 
    """
    coordinates = {}
    # Iterating through the 3D mask
    for i in tqdm(range(masks.shape[0]), position=0, leave=True, desc="\n### Converting mask array to dictionary", ncols=100):
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


def bounding_box(coordinates: dict) -> dict:
    """
    Creates bounding box min- and max-coordinates for every mask.

    Parameter
    ------

    coordinates: dict
        The output dictionary from the function generate_mask_dict().

    Returns:
    ------

    Returns a dictionary with keys being the mask indices and values a list of corresponding bounding box coordinates of shape [z_min, z_max, y_min, y_max, x_min, x_max].
    """

    bboxes = {}
    for index, mask in zip(coordinates.keys(), coordinates.values()):
        bboxes[index] = mask[0].min(), mask[0].max(), mask[1].min(), mask[1].max(), mask[2].min(), mask[2].max()
    return bboxes


def cutoffness_feature(img: np.ndarray, mask_inst: np.ndarray, bbox: list) -> float:
    """
    Calculates the variance of intensity values over the pixels on the perimeter of a nuclear mask.

    Parameter
    ------

    img: np.ndarray
        Image of shape ZYX. Expected to match shape of mask.

    mask: np.ndarray
        Mask array of shape ZYX. Contains only a single mask instance with entries = mask index and 0 otherwise. Expected to match shape of img.
    
    Returns:
    ------

    Scalar value of the variance for this respective mask. The value corresponds to the entire 3D-perimeter of the mask. 
    """

    pixel_values = []
    for z_layer in range(bbox[0], (bbox[1]+1)):
        contour_z = measure.find_contours(mask_inst[z_layer], 0, fully_connected='high')
        for i in range(0, contour_z[0].shape[0]):
            pixel_values.append( img[int(z_layer), int(contour_z[0][i, 0]), int(contour_z[0][i, 1])])
    intensity_variance = np.var(pixel_values)
    return intensity_variance


def main():

    # Calls user parameters from CLI
    args = args_parse()
    image_directory = args.image_directory
    mask_directory = args.mask_directory
    mask_coord_directory = args.mask_dict_path
    filetype = args.filetype
    out_dir = args.out_dir
    
    # Detects all filenames in the user-specified input directories
    img_filenames, mask_filenames, mask_coord_filenames = load_filenames(image_directory, mask_directory, mask_coord_directory, filetype)

    # Creates mask-coordinate .pkl files if missing and subsequently detects their names in the directory
    if len(mask_filenames) != len(mask_coord_filenames):
        print('\n### Could not find a corresponding .pkl file of mask coordinates for every mask file.')
        print('\n### Generating .pkl file of mask coordinates from mask file for every mask file in the directory.')
        for mask_name in mask_filenames:
            masks = imageio.imread(os.path.join(mask_directory, mask_name))
            mask_dict = generate_mask_dict(masks)
            mask_coord_name = 'coordinates_' + str(mask_name[:-4]) + '.pkl'
            save_dir = os.path.join(mask_coord_directory, mask_coord_name)
            with open(rf'{save_dir}', 'wb') as f:
                pickle.dump(mask_dict, f)
            print(f'\n### Saved pickle file of mask coordinates for {mask_name[:-4]} in {mask_coord_directory}.')
        mask_coord_filenames = load_filenames(image_directory, mask_directory, mask_coord_directory, filetype)[2]

    img_filenames.sort()
    mask_filenames.sort()
    mask_coord_filenames.sort()
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
        mask = np.array(mask)
        print(f'\n### Mask array of shape {mask.shape} loaded.')

        assert img.shape == mask.shape, print(f'Error image with name {img_name} and mask with name {mask_name} are of different shape:{img.shape} and {mask.shape}')

        # Loads .pkl file of mask coordinate dictionary
        path = os.path.join(mask_coord_directory, coord_name)
        with open(path, 'rb') as f:
            coordinates = pickle.load(f)
        print(f'\n### Coordinates of {len(coordinates.keys())} masks loaded.')




        feature_matrix = generate_features(img, mask, coordinates)
        csv_name = str(detect_common_name(img_name, mask_name, coord_name)) + '_feature_matrix_big.csv'
        final_outdir = os.path.join(out_dir, csv_name)
        np.savetxt(final_outdir, feature_matrix, delimiter=',',)
        print(f'\n### Created .csv-file for feature-matrix of image {img_name}.')
     
        # Generates and saves mask-centroid distance matrix
        """
        avg_distances = get_avg_neigh_distance(coordinates)[0]
        dist_matrix = get_avg_neigh_distance(coordinates)[1]
        dist_matrix_dir = os.path.join(out_dir, str('dist_matrix_' + detect_common_name(img_name, mask_name, coord_name)))
        np.savetxt(dist_matrix_dir, dist_matrix)
        print(f'\n### Saved the pairwise distance matrix between mask centroids for image \'{img_name}\'.')
        """
        
        
if __name__ == '__main__':
    main()
