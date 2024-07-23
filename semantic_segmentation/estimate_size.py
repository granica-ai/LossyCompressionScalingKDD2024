"""Estimate average pixels per image from Cityscapes dataset files at JXL compression levels."""

import os
from typing import List
import numpy as np
import PIL.Image
from imagecodecs import jpegxl_encode
from tqdm import tqdm
import simplejson as json

from utils.consts import CITYSCAPES_DATA_ROOT, CITYSCAPES_SUBSET_ROOT, FOOD101_SIZES


def calc_jxl_sizes_from_files(
    save_path: str, dist_list: List[int] = None, old_path: str = None
) -> dict:
    """
    Calculate average size of JXL compressed images from 20% of the Cityscapes train images.

    Args:
        save_path (str): Path to save results.
        dist_list (list): List of Butteraugli distances to use as compression levels.
        old_path (str): Optional, path to old results to update.
    """
    subset_file = os.path.join(CITYSCAPES_SUBSET_ROOT, "train0.2.txt")
    data_root = os.path.join(CITYSCAPES_DATA_ROOT, "train")
    data_suffix = "_leftImg8bit.png"
    with open(subset_file, "r") as f:
        subset = f.readlines()
    subset = [x.strip() for x in subset]

    if dist_list is None:
        dist_list = [1, 2, 4, 7, 10, 15]

    if old_path is not None:
        with open(old_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for dist in dist_list:
        print(f"Calculating for distance {dist}")
        total_size = 0
        for img_path in tqdm(subset):
            img = PIL.Image.open(os.path.join(data_root, img_path + data_suffix))
            img = np.array(img)
            num_bits = len(jpegxl_encode(img, distance=dist))
            total_size += num_bits
        results[dist] = total_size / len(subset)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    calc_jxl_sizes_from_files(FOOD101_SIZES)
