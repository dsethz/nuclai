########################################################################################################################
# This script filters out masks that do not fulfill certain criteria. These criteria are measured in the mask_dict and #                                            
# then applied to both the mask_array and the mask_dict.                                                               #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 07.03.2024                                                                                     #                                                                                          #
########################################################################################################################
import numpy as np
import pickle
from itertools import groupby
from tqdm import tqdm
import tifffile
import argparse
import ipdb
from typing import List, Tuple, Dict

parser = argparse.ArgumentParser()

parser.add_argument(
    '--mask_coordinates_path',
    type = str,
    required=True,
    help = 'Filepath where the pickled mask coordinate dictionary is saved. (Including filename and extension).'
)
parser.add_argument(
    '--mask_array_path',
    type = str,
    required=True,
    help = 'Filepath where the mask array is saved. (Including filename and extension).'
)
parser.add_argument(
    '--save_dir_pkl',
    type = str,
    required=True,
    help = 'Directory where the filtered mask coordinate dictionary should be saved. (Including filename and extension).'
)
parser.add_argument(
    '--save_dir_tif',
    type = str,
    required=True,
    help = 'Directory where the filtered mask array should be saved. (Including filename and extension).'
)

args =  parser.parse_args()
mask_coordinates_path = args.mask_coordinates_path
mask_array_path = args.mask_array_path
save_dir_pkl = args.save_dir_pkl
save_dir_tif = args.save_dir_tif

def reconstruct_arr_from_dict(arr_shape, coordinates):
    filtered_mask_array = np.zeros(arr_shape)
    for key in tqdm(coordinates.keys(), desc='\n### Reconstructing mask.tif file with filtered mask-coordinates .pkl file.'):
        for i in range(0, coordinates[key][0].shape[0]):
            filtered_mask_array[coordinates[key][0][i], coordinates[key][1][i], coordinates[key][2][i]] = key
    return filtered_mask_array

# Loads the mask array
mask_array = tifffile.imread(mask_array_path)
mask_shape = mask_array.shape
print(f'\n### Mask of shape {mask_array.shape} loaded.')

# Loads the mask coordinate dict
with open(mask_coordinates_path, 'rb') as f:
    coordinates = pickle.load(f)
print(f'\n### Successfully loaded pickled mask coordinate file containing {int(len(coordinates))} masks.')

#remove the zero-key in the dictionary
if (0 in coordinates.keys()) == True:
    coordinates.pop(0)

#filters are applied to the coordinates.pkl and then transferred to the mask array tif
keys_to_keep = set([])
keys_to_pop = set([])


#Filtering criterion 1: Masks must appear in 2 or more z-layers.
for key in tqdm(coordinates.keys(), desc='\n### Filtering masks for only appearing in on z-layer:'):
    z_list = coordinates[key][0]
    if len(np.unique(z_list)) > 1:
        keys_to_keep.add(key)
keys_to_pop = []
keys_to_pop = [k for k in coordinates.keys() if k not in keys_to_keep]
print(f'\n### Detected {len(keys_to_keep)} correct masks.')
deleted_keys = keys_to_pop ################
for key_to_pop in keys_to_pop:
    del coordinates[key_to_pop]
print(f'\n### Removed {len(keys_to_pop)} masks that only appear on one z_layer. Mask_ids: {keys_to_pop}')


#Filtering criterion 2: Masks must be a surface on XY (E.g. not be a line.).
keys_to_keep = set([])
keys_to_pop = set([])
for key in tqdm(coordinates.keys(), desc='\n### Filtering masks for missing volume:'):
    x_list = coordinates[key][2].copy()
    x_list.sort()
    x_counts = [sum(1 for _ in group) for _, group in groupby(x_list)]
    for i in range(len(x_counts)-1):
        if x_counts[i] and x_counts[i+1] > 3:
            keys_to_keep.add(key)
keys_to_pop = [k for k in coordinates.keys() if k not in keys_to_keep]
deleted_keys = deleted_keys + list(keys_to_pop) ###################
print(f'\n### Detected {len(keys_to_keep)} correct masks.')

for key_to_pop in keys_to_pop:
    del coordinates[key_to_pop]
print(f'\n### Removed {len(keys_to_pop)} masks that do not have proper volume. Mask_ids: {keys_to_pop}')


#Filtering criterion 3: Masks must have a total voxel count between 25 and 75000 voxels.
keys_to_keep = set([])
keys_to_pop = set([])
min_volume = 25
max_volume = 75000
for key in tqdm(coordinates.keys(), desc='\n### Filtering masks by absolute voxel count (volume):'):
    x_list = coordinates[key][2]
    if len(x_list) > min_volume and len(x_list) < max_volume:
        keys_to_keep.add(key)
keys_to_pop = [k for k in coordinates.keys() if k not in keys_to_keep]
print(f'\n### Detected {len(keys_to_keep)} correct masks.')
deleted_keys = deleted_keys + list(keys_to_pop) #####################
for key_to_pop in keys_to_pop:
    del coordinates[key_to_pop]
print(f'\n### Removed {len(keys_to_pop)} masks that have too high or too low volume. Mask_ids: {keys_to_pop}')

# Saving filtered mask coordinates.pkl file
with open(rf'{save_dir_pkl}', 'wb') as f:
    pickle.dump(coordinates, f)
print(f'\n### Saved pickle file of filtered mask coordinates.')

# The filtered dictionary is used to create the filtered mask array.
reconstructed_post_filtering = reconstruct_arr_from_dict(mask_shape, coordinates)
tifffile.imwrite(save_dir_tif, reconstructed_post_filtering)
print(f'\n### Saved .tif file of filtered masks.')