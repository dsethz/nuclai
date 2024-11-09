########################################################################################################################
# This script creates a matrix of 3D features for every single mask.                                                   #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.10.13                                                                                        #
# Date:                 07.03.2024                                                                                     #
########################################################################################################################
import argparse
import os

import pandas as pd

# import imageio.v2 as imageio
import tifffile
from skimage import measure


def args_parse():
    """
     Catches user input from the CLI.

    Parameters

    ----------

    -

    Return

    ------

    Returns a namespace from `argparse.parse_args()`.

    """
    desc = "Program to generate feature matrices for segmented nuclei."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--image",
        type=str,
        default=r"N:\schroeder\Data\DS\PhD\nucleus_classification\raw_data\3D\Alphacat_rep2\stacked\subset\c0_0-55_1950-4500_11550-14862.tif",
        help=("Path to nucleus image."),
    )

    parser.add_argument(
        "--mask",
        type=str,
        default=r"N:\schroeder\Data\DS\PhD\nucleus_classification\data\3d\images_and_features\segmentation\mouse\cd41\acat_rep2\subset\5_curated_filtered\c0_0-55_1950-4500_11550-14862_cp_masks_comb.tif",
        help=("Path to segmentation mask"),
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 0.24, 0.24),
        help=("Voxel spacing for the images."),
    )

    parser.add_argument(
        "--out",
        type=str,
        default=r"N:\schroeder\Data\DS\PhD\nucleus_classification\data\3d\images_and_features\segmentation\mouse\cd41\acat_rep2\subset\7_classic_features",
        help="Path to output directory.",
    )

    return parser.parse_args()


def main():
    args = args_parse()
    path_i = args.image
    path_m = args.mask
    spacing = tuple(args.spacing)
    path_out = args.out

    os.makedirs(path_out, exist_ok=True)

    # select properties
    properties = [
        "area",  # Area of the region i.e. number of pixels of the region scaled by pixel-area.
        "area_bbox",  # Area of the bounding box i.e. number of pixels of the bounding box scaled by pixel-area.
        "area_convex",  # Area of the convex hull image, the smallest convex polygon enclosing the region.
        "area_filled",  # Area of the region with all holes filled in.
        "axis_major_length",  # Length of the major axis of the ellipse that matches the second central moments of the region.
        "axis_minor_length",  # Length of the minor axis of the ellipse that matches the second central moments of the region.
        #    "bbox",  # Bounding box (min_row, min_col, max_row, max_col) of the region.
        #    "centroid",  # Centroid coordinate tuple (row, col) of the region.
        #    "centroid_local",  # Centroid coordinate tuple (row, col) relative to the region bounding box.
        #    "centroid_weighted",  # Centroid weighted with intensity values, giving (row, col) coordinates.
        #    "centroid_weighted_local",  # Intensity-weighted centroid relative to the region bounding box.
        #    "coords_scaled",  # Coordinates of the region scaled by spacing, in (row, col) format.
        #    "coords",  # Coordinates of the region in (row, col) format.
        "eccentricity",  # Eccentricity of the ellipse matching the second moments of the region (range [0, 1), with 0 being circular).
        "equivalent_diameter_area",  # Diameter of a circle with the same area as the region.
        "euler_number",  # Euler characteristic: number of connected components minus the number of holes in the region.
        "extent",  # Ratio of the region’s area to the area of the bounding box (area / (rows * cols)).
        "feret_diameter_max",  # Maximum Feret's diameter: longest distance between points on the convex hull of the region.
        #    "image",  # Binary region image, same size as the bounding box.
        #    "image_convex",  # Binary convex hull image, same size as the bounding box.
        #    "image_filled",  # Binary region image with holes filled, same size as the bounding box.
        #    "image_intensity",  # Intensity image inside the region's bounding box.
        "inertia_tensor",  # Inertia tensor for rotation around the region's center of mass.
        "inertia_tensor_eigvals",  # Eigenvalues of the inertia tensor, in decreasing order.
        "intensity_max",  # Maximum intensity value in the region.
        "intensity_mean",  # Mean intensity value in the region.
        "intensity_min",  # Minimum intensity value in the region.
        # "intensity_std",  # Standard deviation of the intensity values in the region. # Doesnt work.
        "label",  # Label of the region in the labeled input image.
        "moments",  # Spatial moments up to the 3rd order.
        "moments_central",  # Central moments (translation-invariant) up to the 3rd order.
        "moments_hu",  # Hu moments (translation, scale, and rotation-invariant).
        "moments_normalized",  # Normalized moments (translation and scale-invariant) up to the 3rd order.
        "moments_weighted",  # Intensity-weighted spatial moments up to the 3rd order.
        "moments_weighted_central",  # Intensity-weighted central moments (translation-invariant) up to the 3rd order.
        "moments_weighted_hu",  # Intensity-weighted Hu moments (translation, scale, and rotation-invariant).
        "moments_weighted_normalized",  # Intensity-weighted normalized moments up to the 3rd order.
        "num_pixels",  # Number of foreground pixels in the region.
        "orientation",  # Orientation of the major axis of the ellipse that matches the second moments of the region.
        "perimeter",  # Perimeter of the object, approximated using a 4-connectivity contour.
        "perimeter_crofton",  # Perimeter estimated by the Crofton formula, based on 4 directions.
        #    "slice",  # Slice object to extract the region from the source image.
        "solidity",  # Solidity: ratio of region area to convex hull area.
    ]  # The removed properties are not informative as they relate to absolute positions in the image

    # load image/mask/coords
    img_name = os.path.basename(path_i).split(".")[0]
    image = tifffile.imread(path_i)
    mask = tifffile.imread(path_m)

    # get features
    mask_3D_features = {}
    for prop in properties:
        try:
            mask_3D_prop = measure.regionprops_table(
                mask,
                intensity_image=image,
                spacing=spacing,
                properties=[prop],
            )
            for key, value in mask_3D_prop.items():
                mask_3D_features[key] = value
        except Exception as e:  # noqa BLE001
            print(f"Error calculating 3D property {prop}: {e}")

    features_3D = pd.DataFrame(mask_3D_features)
    features_3D.rename(columns={"label": "mask_id"}, inplace=True)
    features_3D.to_csv(
        os.path.join(path_out, f"classic_features_3D_{img_name}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
