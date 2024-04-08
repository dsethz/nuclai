########################################################################################################################
# This script stacks tif files of multiple channels and z-slices into a single tif-hyperstack of shape ZCYX.           #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 07.03.2024                                                                                     #                                                                                          #
########################################################################################################################
import glob
import os
import re
import argparse

import numpy as np 

from skimage import io
from tqdm import tqdm


desc = 'This program stacks multiple tif files to a z-stack'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    '--in_dir',
    type=str,
    required=True,
    help='The directory containing the individual .tif files. Tif files mus have identical shape in each z-sclice. Filename must contain channel as <C1>',
)
parser.add_argument(
    '--out_dir',
    type=str,
    required=True,
    help='Directory to save the output stacked files.'
)


args = parser.parse_args()
input_dir = args.in_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

#creating a list of channel names
channels = []
pattern = re.compile(r'_C(\d+)_Z(\d+)')
files = glob.glob(os.path.join(input_dir, '*.tif'))
files.sort()
prefix = os.path.commonprefix(files)
prefix = prefix.split(os.sep)[-1]
c_combinations = []
z_layers = []

for f in files:
    try:
        c, z = re.search(pattern, f).groups()
        c_combinations.append(c)
        if c == '1' or c == '01':
            z_layers.append(z)
    except:
        print(f"Error: {f} does not match the pattern.")
        continue

channels = list(set(c_combinations))
channels.sort()
print('\n### Channels', channels, "detected.")

#initialize multichannel z-stack as np array
first_image = io.imread(files[0])

# Gives information about datatype of a single slice
print('\n### Shape:', first_image.shape)
print('\n### Data_type:', first_image.dtype)
print('\n### Memory usage in bytes:', first_image.nbytes)
print('\n### Array min and max value:', first_image.min(), first_image.max())

#read in the individual .tif files into the np array
for channel in channels:
    print(f'\n### Reading channel {channel}')
    tmp = np.zeros((len(z_layers), first_image.shape[0], first_image.shape[1]), dtype=first_image.dtype)
    tif_dir_list = []
    tif_dir_list = [s for s in files if f'_C{channel}_' in s]
    tif_dir_list.sort()

    for f in tqdm(tif_dir_list):
        z_slice = io.imread(f)
        tmp[tif_dir_list.index(f), :, :] = z_slice

    # Gives information about datatype of the output hyperstack
    print(f'\n### Channel {channel} shape:', tmp.shape)
    print(f'\n### Channel {channel} data_type:', tmp.dtype)
    print(f'\n### Channel {channel} memory usage in bytes:', tmp.nbytes)
    print(f'\n### Channel {channel} min and max value:', tmp.min(), tmp.max())

    #writing the numpy array to the final output tifffile
    io.imsave(os.path.join(out_dir, f'{prefix}{channel}.tif'), tmp)

    print(f"\n### Saved Channel {channel} with shape, {tmp.shape} and saved it in {out_dir}.")
