"""Estimate average pixels per image from iSAID dataset files at JXL compression levels."""

import os
import numpy as np
import glob
import PIL.Image
from imagecodecs import jpegxl_encode
from tqdm import tqdm
import simplejson as json

from utils.consts import ISAID_DATA_ROOT, ISAID_SIZES


def calc_jxl_sizes_from_files(
    save_path: str, dist_list: list = None, old_path: list = None
) -> dict:
    """
    Calculate average size of JXL compressed images from 1000 random iSAID images.

    Args:
        save_path (str): Path to save results.
        dist_list (list): List of Butteraugli distances to use as compression levels.
        old_path (str): Path to old results to update.
    """
    img_files = glob.glob(os.path.join(ISAID_DATA_ROOT, "train", "images", "*.png"))
    labels = glob.glob(
        os.path.join(ISAID_DATA_ROOT, "train", "images", "*instance*.png")
    )
    img_files = set(img_files) - set(labels)
    rng = np.random.default_rng(42)
    subset = rng.choice(list(img_files), 1000, replace=False)
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
            img = np.array(PIL.Image.open(img_path))
            num_bits = len(jpegxl_encode(img, distance=dist))
            total_size += num_bits
        results[dist] = total_size / len(subset)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    calc_jxl_sizes_from_files(save_path=ISAID_SIZES)
