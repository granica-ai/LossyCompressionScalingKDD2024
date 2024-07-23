"""Get the uncompressed subset for each storage size for the Cityscapes dataset."""

import os
import logging
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

from semantic_segmentation.curves.make_deterministic_subsets import (
    get_cityscapes_paths,
)
from utils.consts import CITYSCAPES_DATA_ROOT, CITYSCAPES_PATH, CITYSCAPES_SUBSET_ROOT
from utils.expt_utils import make_logger
from utils.randomized_curve_utils import get_dataset_size


def set_up_logging(expt_name: str):
    """
    Set up logging for finding uncompressed subsets.

    Args:
        expt_name (str): The name of the experiment.
    """
    log_file = os.path.join(
        CITYSCAPES_PATH, expt_name, "subset_file_creation_{date}.log"
    )
    make_logger(log_file)


def estimate_size_per_img(ds: List[str], ba: float, subsample_size: int = 500) -> float:
    """
    Estimate the size of a single image at a given Butteraugli distance.

    Args:
        ds (List[str]): The list of image paths.
        ba (float): The Butteraugli distance.
        subsample_size (int): The size of the subsample to use for estimation.

    Returns:
        float: The estimated size of a single image.
    """
    subsample, _ = train_test_split(ds, train_size=min(len(ds), subsample_size))
    images = get_full_paths(subsample)
    dist_map = dict(zip(images, [ba] * len(images)))
    subsample_size = get_dataset_size(distance_map=dist_map, ds=images)
    return subsample_size / len(subsample)


def get_images_for_target_size(
    current_subset: List[str],
    target_size: int,
    ba: float,
    epsilon: float = 0.001,
    random_state: int = 42,
) -> List[str]:
    """
    Get a subset of images that will compress to close to a target size at a given Butteraugli distance.

    Args:
        current_subset (List[str]): The list of image paths.
        target_size (int): The target size of the subset.
        ba (float): The Butteraugli distance.
        epsilon (float): The error tolerance.
        random_state (int): The random seed.

    Returns:
        List[str]: The subset of image paths.
    """
    logging.info(
        f"\n====== Trying to find images for target size {target_size} at ba {ba} ======"
    )
    current_error = 1
    size_per_img = estimate_size_per_img(current_subset, ba)
    num_images = int(target_size / size_per_img)
    seen_sizes = set()
    while current_error > epsilon:
        logging.info(f"Estimated num images: {num_images}")
        subsample, _ = train_test_split(
            current_subset,
            train_size=num_images,
            random_state=random_state,
        )
        if len(subsample) in seen_sizes:
            logging.warning(
                f"The search did not fully converge, size is {len(subsample)} and current error is {current_error}."
            )
            break
        seen_sizes.add(len(subsample))
        logging.info(f"Subsample count: {len(subsample)}")

        subsample_size = get_dataset_size(distance=ba, ds=get_full_paths(subsample))
        logging.info(f"Subsample size: {subsample_size}")

        current_error = abs(subsample_size - target_size) / target_size
        logging.info(f"Current error: {current_error}")
        correction = (target_size - subsample_size) / (subsample_size / len(subsample))
        num_images = len(subsample) + int(correction)
    return subsample


def get_full_paths(subsample: List[str]) -> List[str]:
    """
    Get the full paths for a list of image filenames.

    Args:
        subsample (List[str]): The list of image filenames.

    Returns:
        List[str]: The list of full image paths.
    """
    data_suffix = "_leftImg8bit.png"
    full_paths = [
        os.path.join(CITYSCAPES_DATA_ROOT, "train", x.strip() + data_suffix)
        for x in subsample
    ]
    return full_paths


def make_uncompressed_subset_files(random_state: int = 42):
    """
    Make uncompressed subset files for a range of target sizes.

    Args:
        random_state (int): The random seed.
    """
    set_up_logging("uncompressed")
    ds = get_cityscapes_paths()
    ss = [int(2000 * 60600 * x) for x in np.linspace(0.3, 0.99, 5)]
    for s in ss:
        subset = get_images_for_target_size(ds, s, ba=0, random_state=random_state)
        suffix = f"_seed{random_state}" if random_state != 42 else ""
        subset_file = os.path.join(
            CITYSCAPES_SUBSET_ROOT, "uncompressed", f"uncompressed_s{s}{suffix}.txt"
        )
        with open(subset_file, "w") as f:
            f.writelines([x + "\n" for x in subset])


if __name__ == "__main__":
    make_uncompressed_subset_files()
    make_uncompressed_subset_files(random_state=0)
    make_uncompressed_subset_files(random_state=1)
