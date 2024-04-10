########################################################################################################################
# This script generates a memory and computational efficient representation of the mask-array. This representation is  #
# a dictionary, in which the keys are the index of the mask and the values a tuple of 3 arrays. The arrays represent   #
# the Z, Y and X components of all pixels within the mask.                                                             #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 09.04.2024                                                                                     #                                                                                          #
########################################################################################################################
import tifffile
import numpy as np
import pickle
from itertools import groupby
from tqdm import tqdm
import ipdb
import argparse
import gc
import shelve
import os
import sys
from typing import List, Tuple, Dict
import pstats

desc = 'Generates a more efficient mask coordinates dictionary representation of the mask array.'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    '--mask_array_path',
    type = str,
    required = True,
    help = 'Path to the mask array.'
)   
parser.add_argument(
    '--save_dir',
    type = str,
    required = True,
    help = 'Filepath where mask dictionary should be saved, without filename and .pkl extension' 
)
parser.add_argument(
    '--save_name',
    type = str,
    required = True,
    help = 'Name of file that should be saved without extension.' 
)
parser.add_argument(
    '--max_keys',
    type = int,
    default=100000,
    help = 'Maximum number of keys the dictionary should hold for memory purposes.' 
)

args = parser.parse_args()
mask_array_path = args.mask_array_path
save_dir = args.save_dir
max_keys = args.max_keys
save_name = args.save_name

# Loading mask as uint32 np.array
print(f'\n### Received {mask_array_path} as input path and {save_dir} as save directory.')
mask_array_float = tifffile.imread(mask_array_path)
mask_shape = mask_array_float.shape
print(f'\n### Mask of shape {mask_array_float.shape} loaded.')
mask_array = mask_array_float.astype(np.uint32)
del mask_array_float

# Determining all mask values and correspondin voxel count for dictionary initialization
values, counts = np.unique(mask_array, return_counts=True)
values = values[1:]
counts = counts[1:]
print('\n### Mask sizes obtained.')

# When creating multiple dicts (for memory efficiency), partition keys.
def partition_keys(max_keys: int, all_keys: np.array) -> List[Tuple[int]]:
    """
    When creating multiple dicts for working memory efficiency, the keys are partitioned such as that the dictionaries contain
    a maximum of max_keys keys. 

    Parameter
    ------

    max_keys: int
        Number of keys that should be maximally in the dictionary

    all_keys: np.array
        Array holding all unique values of the mask array e.g. all mask indices / all keys.

    Returns:
    ------

    Returns a list with n tuples of (min_key, max_key) for n dictionaries of size max_keys to be created. 
    """
    key_bounds_list = []
    for i in range(0, len(all_keys), max_keys):
        key_bounds = (min(all_keys[i:i+max_keys]), max(all_keys[i:i+max_keys]))
        key_bounds_list.append(key_bounds)
    print('\n### Partitionings created.')
    return key_bounds_list


def mask_mask_array(mask_array: np.array, key_min: int, key_max: int) -> np.array:
    """
    Creates an array of same size as the mask-array where all masks that have indices outside the partitioning 
    of min_key, max_key are removed.

    Parameter
    ------

    key_min: int
        Lowest mask index to be in the mask array.

    key_max: int
        Highest mask index to be in the mask array.

    Returns:
    ------

    Returns a copy of the input mask array, where all masks outside the partitioning are removed. 
    """
    mask_array2 = mask_array.copy()
    mask_array2[mask_array2 < key_min] = 0
    mask_array2[mask_array2 > key_max] = 0
    print('\n### Masked mask array created.')
    return mask_array2





def generate_mask_dict(masks: np.ndarray, values: np.array, counts: np.array) -> Dict[int, Tuple[np.array, np.array, np.array]]: 
    """
    Function that converts the mask array into a more memory and computationally efficient dictionary representation.

    Parameter
    ------

    masks: np.ndarray
        The mask array. Same size as nuclei image. The value in the voxels corresponds to the mask-idx, 0 indicates background voxels.

    values: np.ndarray
        A np.array containg all unique values (i.e. mask indices) of the mask-array.

    Returns:
    ------

    Returns a dictionary where keys are the indices of the masks and values are their voxel coordinates. Their voxel coordinates are represented as 
    a tuple of three np.arrays of same size. The first array holds the Z components of the voxels, the second Y-components and the third X-components.
    Example to obtain the coordinates of the first voxel (Z, Y, X) one would have to index (coordinates[key][0][0], coordinates[key][1][0], coordinates[key][2][0]) 
    """
    # Initializing dict that holds tuple of three np.arrays of correct length for each mask (one entry excess for efficient writing from left to right)
    coordinates = {}
    for value, count in zip(values, counts):
        coordinates[value] = (np.zeros(count, dtype=np.uint32), np.zeros(count, dtype=np.uint32), np.zeros(count, dtype=np.uint32))
    
    # Memorizing for each key, where the next coordinate should be written within the array
    cursor = {}
    for key in coordinates.keys():
        cursor[key] = 0
    
    masks_shape = masks.shape
    for z in tqdm(range(mask_shape[0]), desc='Generating a mask dict.', ncols=100):
        for y in range(masks_shape[1]):
            for x in range(masks_shape[2]):
                # Value of voxel (mask-id) in mask array
                voxel_value = int(masks[z, y, x])
                
                if voxel_value != 0:
                    # Write the zyx position in the npy aray in the dict value
                    coordinates[voxel_value][0][cursor[voxel_value]] = z
                    coordinates[voxel_value][1][cursor[voxel_value]] = y
                    coordinates[voxel_value][2][cursor[voxel_value]] = x
                    # increment the cursor for the mask 
                    cursor[voxel_value] += 1

    return dict(coordinates)





def wrapper():
    key_bounds_list = partition_keys(max_keys, values)
    for i, bounds in enumerate(key_bounds_list):
        print(f'Starting iteration {i} with a maximum of {max_keys} masks.')
        masked_mask_array = mask_mask_array(mask_array, bounds[0], bounds[1]) 
        coordinates = generate_mask_dict(masked_mask_array, values, counts)
        filename = str(save_name + f'_part_{i+1}.pkl')
        save_dir_final = os.path.join(save_dir, filename)
        with open(rf'{save_dir_final}', 'wb') as f:
            pickle.dump(coordinates, f)
            print(f'\n### Saved pickle file of mask coordinates.')
        del masked_mask_array, coordinates

wrapper()


