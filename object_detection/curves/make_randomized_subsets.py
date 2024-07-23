"""Find the randomized compression levels to achieve a target size for the iSAID dataset."""

import os
import logging
import datetime
import glob
import numpy as np
from scipy.stats import rankdata
import simplejson as json

from utils.consts import ISAID_DATA_ROOT, ISAID_PATH
from utils.expt_utils import binary_search_largest_below_function, make_logger
from utils.randomized_curve_utils import get_compression_fn, get_dataset_size


def set_up_logging(min_compression: float, max_compression: float):
    """
    Set up logging for finding randomized subsets.

    Args:
        min_compression (float): The minimum compression level.
        max_compression (float): The maximum compression level.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(
        ISAID_PATH,
        "randomized",
        f"make_dist_maps_min{min_compression}_max{max_compression}_{date}.log",
    )
    make_logger(log_file)


def get_subset_json(target_s: int, random_seed: int = 42) -> dict:
    """
    Load a subset json file for a given target size and random seed.

    Args:
        target_s (int): The target size of the subset.
        random_seed (int): The random seed used to create the subset.

    Returns:
        dict: The subset json.
    """
    suffix = f"_seed{random_seed}" if random_seed != 42 else ""
    subset_file = glob.glob(
        os.path.join(
            ISAID_PATH, "deterministic", f"idxs_s{target_s}_n[1-9]*{suffix}.json"
        )
    )[0]
    logging.info(f"Loading subset from {subset_file}")
    with open(subset_file) as f:
        subset = json.load(f)

    return subset


def make_distance_map_file(
    min_compression: float, max_compression: float, target_s: int, random_seed: int = 42
):
    """
    Find a randomized compression distance mapping to make a given subset compress
    to close to a target size.

    Args:
        min_compression (float): The minimum compression level Butteraugli distance.
        max_compression (float): The maximum compression level Butteraugli distance.
        target_s (int): The target size of the subset in bytes.
        random_seed (int): The random seed used to determine randomized compression.
    """
    logging.info(f"\n====== Target size: {target_s} ======\n")
    subset = get_subset_json(target_s, random_seed)
    images = [example["file_name"] for example in subset["images"]]
    images = [
        os.path.join(ISAID_DATA_ROOT, "train/images", img_file) for img_file in images
    ]
    np.random.seed(random_seed)
    fake_score = rankdata(np.random.rand(len(images)))

    compression_fn, normalization = get_compression_fn(
        images,
        target_s,
        min_compression,
        max_compression,
        fake_score,
    )

    def ds_size_fn(dist):
        args = {f"{normalization}_dist": -dist, "paths": images}
        compression_level = compression_fn(**args)
        logging.info(f"Compression level min: {min(compression_level.values())}")
        logging.info(f"Compression level max: {max(compression_level.values())}")
        size = get_dataset_size(compression_level, images)
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

    save_path = os.path.join(
        ISAID_PATH,
        "randomized",
        f"shuffle_{target_s}_min{min_compression}_max{max_compression}_seed{random_seed}_compression_levels.json",
    )
    logging.info(f"Saving to {save_path}")
    compression_level = compression_fn(
        **{f"{normalization}_dist": -opt_minmax}, paths=images
    )
    with open(save_path, "w") as f:
        json.dump(compression_level, f, indent=4)


if __name__ == "__main__":
    min_compression = 0
    max_compression = 15
    set_up_logging(min_compression, max_compression)
    ss = [84087000, 168174000, 252261000, 336348000, 420435000]
    logging.info(f"Using s values: {ss}")
    for seed in [42, 0, 1]:
        for target_s in ss:
            make_distance_map_file(
                min_compression, max_compression, target_s, random_seed=seed
            )
