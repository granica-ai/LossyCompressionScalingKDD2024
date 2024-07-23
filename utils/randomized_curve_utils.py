"""Helper functions for randomized curve generation."""

import logging
import functools
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from utils.drr_estimation_tools import image_size_at_dist


def get_comp_levels(
    min_dist: float, max_dist: float, fake_score: np.ndarray, paths: List[str]
) -> dict:
    """
    Get compression levels uniformly distributed between a min and max distance according to a random ordering.

    Args:
        min_dist (float): The minimum distance.
        max_dist (float): The maximum distance.
        fake_score (np.ndarray): The fake score for each image.
        paths (List[str]): The paths to the images.

    Returns:
        dict: The compression levels for each image.
    """
    cls = (1 - fake_score / np.max(fake_score)) * (max_dist - min_dist) + min_dist
    cls[cls < 0] = 0
    return dict(zip(paths, cls))


def get_compression_fn(
    subset: dict,
    target_s: int,
    min_compression: float,
    max_compression: float,
    fake_score: np.ndarray,
) -> Tuple[callable, str]:
    """
    Get a function that uses either the min or max compression level as input to determine
    compression levels for each image.

    If the maximum compression leads to insufficient compression, increase the minimum
    compression level to achieve higher DRR.

    If the maximum compression leads to too much compression, reduce the maximum compression
    level to achieve lower DRR.

    Args:
        subset (dict): The subset json.
        target_s (int): The target size of the subset in bytes.
        min_compression (float): The minimum compression level Butteraugli distance.
        max_compression (float): The maximum compression level Butteraugli distance.
        fake_score (np.ndarray): The fake score for each image, used to order the images.

    Returns:
        Tuple[callable, str]: The compression function and the normalization used.
    """
    max_max = get_comp_levels(min_compression, max_compression, fake_score, subset)
    max_max_size = get_dataset_size(max_max, subset)
    if max_max_size > target_s:
        logging.info(f"Using min compression: {max_max_size} for {target_s}")
        compression_fn = functools.partial(
            get_comp_levels,
            max_dist=max_compression,
            fake_score=fake_score,
        )
        normalization = "min"
    else:
        logging.info(f"Using max compression: {max_max_size} for {target_s}")
        compression_fn = functools.partial(
            get_comp_levels,
            min_dist=min_compression,
            fake_score=fake_score,
        )
        normalization = "max"
    return compression_fn, normalization


def get_dataset_size(distance_map: dict, ds: list) -> int:
    """
    Get the size of a dataset given a distance map and a dataset.

    Args:
        distance_map (dict): A map from image file paths to compression levels.
        ds (list): The dataset, a list of image file paths.

    Returns:
        int: The size of the dataset in bytes.
    """
    func = functools.partial(get_image_size, distance_map=distance_map)
    with Pool(10) as p:
        sizes = list(tqdm(p.imap(func, ds), total=len(ds)))
    logging.debug(f"Sizes: {sizes}")
    return sum(sizes)


def get_image_size(img_path: str, distance_map: dict) -> int:
    """
    Get the size of an image given a distance map.

    Args:
        img_path (str): The image file path.
        distance_map (dict): A map from image file paths to compression levels.

    Returns:
        int: The compressed size of the image in bytes.
    """
    distance = distance_map[img_path]
    return image_size_at_dist(img_path, distance)
