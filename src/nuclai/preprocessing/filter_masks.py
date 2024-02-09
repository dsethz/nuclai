########################################################################################################################
# This script filters out masks that do not fulfill certain criteria-                                                  #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# PyTorch Version:      2.1.2                                                                                          #
# Lightning Version:    2.1.3                                                                                          #
########################################################################################################################

import numpy as np
import pickle
from itertools import groupby
from tqdm import tqdm
import ipdb
import tifffile


mask_coordinates_path = r'N:\schroeder\Data\LR\feature_test\mask_dict\coordinates_FB22.pkl'

mask_array_path = r'N:\schroeder\Data\LR\feature_test\mask\FB22_63x_C0_cp_masks.tif'

mask_array = tifffile.imread(mask_array_path)
mask_shape = mask_array.shape()

with open(mask_coordinates_path, 'rb') as f:
    coordinates = pickle.load(f)
print(f'Loaded {len(coordinates.keys())} masks.')

#filters are applied to the coordinates.pkl and then transferred 
#Filter out masks that dont have a volume, but are only a line
keys_to_keep = set([])
for key in tqdm(coordinates.keys(), desc='\n### Filtering masks:'):
    x_list = coordinates[key][2]
    x_list.sort()
    x_counts = [sum(1 for _ in group) for _, group in groupby(x_list)]
    for i in range(len(x_counts)-1):
        if x_counts[i] and x_counts[i+1] > 3:
            keys_to_keep.add(key)
print(f'\n### Detected {len(keys_to_keep)} correct masks.')
keys_to_pop = [k for k in coordinates.keys() if k not in keys_to_keep]
print(f'\n### Detected {len(keys_to_pop)} masks without proper volume.')
ipdb.set_trace()
for key in keys_to_pop:
    coordinates.pop(key)
print(f'Masks with indeces {keys_to_pop} removed from coordinates, because they contains no volume along XY.')

mask_array = np.zeros(mask_shape)

for key in tqdm(coordinates.keys(), desc='Writing mask array'):
    mask_array[coordinates[key][0], coordinates[key][1], coordinates[key][2]] = int(key)


    
    