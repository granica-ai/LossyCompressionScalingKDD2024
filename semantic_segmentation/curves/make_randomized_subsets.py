"""Find the randomized compression levels to achieve a target size for the Cityscapes dataset."""

import os
import logging
import datetime
import glob
from typing import List
import numpy as np
from scipy.stats import rankdata
import simplejson as json

from utils.consts import CITYSCAPES_DATA_ROOT, CITYSCAPES_PATH, CITYSCAPES_SS
from utils.expt_utils import binary_search_largest_below_function, make_logger
from utils.randomized_curve_utils import get_compression_fn, get_dataset_size


def set_up_logging(min_compression: float, max_compression: float):
    """
    Set up logging for finding randomized subsets.

    Args:
        min_compression (float): The minimum allowed compression level.
        max_compression (float): The maximum allowed compression level.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(
        CITYSCAPES_PATH,
        "randomized",
        f"make_dist_maps_min{min_compression}_max{max_compression}_{date}.log",
    )
    make_logger(log_file)


def get_full_path_subset(target_s: int, random_seed: int = 42) -> List[str]:
    """
    Load a subset as full paths for a given target size and random seed.

    Args:
        target_s (int): The target size of the subset.
        random_seed (int): The random seed used to create the subset.

    Returns:
        List[str]: The subset as full paths.
    """
    suffix = f"_seed{random_seed}" if random_seed != 42 else ""
    subset_file = glob.glob(
        os.path.join(
            CITYSCAPES_PATH, "deterministic", f"idxs_s{target_s}_n[1-9]*{suffix}.npy"
        )
    )[0]
    logging.info(f"Loading subset from {subset_file}")
    with open(subset_file) as f:
        subset = f.readlines()

    subset = [
        os.path.join(CITYSCAPES_DATA_ROOT, "train", x.strip() + "_leftImg8bit.png")
        for x in subset
    ]

    return subset


def make_distance_map_file(
    min_compression: float, max_compression: float, target_s: int, random_seed: int = 42
):
    """
    Make a randomized compression level map to achieve a target storage size.

    Args:
        min_compression (float): The minimum allowed compression level.
        max_compression (float): The maximum allowed compression level.
        target_s (int): The target size of the subset.
        random_seed (int): The random seed used to create the subset.
    """
    logging.info(f"\n====== Target size: {target_s} ======\n")
    subset = get_full_path_subset(target_s, random_seed)
    np.random.seed(random_seed)
    fake_score = rankdata(np.random.rand(len(subset)))

    compression_fn, normalization = get_compression_fn(
        subset,
        target_s,
        min_compression,
        max_compression,
        fake_score,
    )

    def ds_size_fn(dist):
        args = {f"{normalization}_dist": -dist, "paths": subset}
        compression_level = compression_fn(**args)
        size = get_dataset_size(compression_level, subset)
        logging.info(f"Size: {size}")
        return size

    logging.info("Starting binary search")
    opt_minmax, result_size = binary_search_largest_below_function(
        ds_size_fn,
        target_s,
        left=-max_compression,
        right=-min_compression,
        precision=1e-3,
    )
    logging.info(f"Opt min or max: {opt_minmax}")
    logging.info(f"Result size: {result_size}")
    logging.info(f"Error: {(result_size - target_s) / target_s:.2%}")

    save_path = f"/mnt/aurora/km/models/cityscapes/scaling/randomized/shuffle_{target_s}_min{min_compression}_max{max_compression}_seed{random_seed}_compression_levels.json"
    logging.info(f"Saving to {save_path}")
    compression_level = compression_fn(
        **{f"{normalization}_dist": -opt_minmax}, paths=subset
    )
    with open(save_path, "w") as f:
        json.dump(compression_level, f, indent=4)


if __name__ == "__main__":
    min_compression = 0
    max_compression = 15
    set_up_logging(min_compression, max_compression)
    logging.info(f"Using s values: {CITYSCAPES_SS}")
    for target_s in CITYSCAPES_SS:
        for seed in [42, 0, 1]:
            make_distance_map_file(
                min_compression, max_compression, target_s, random_seed=seed
            )
