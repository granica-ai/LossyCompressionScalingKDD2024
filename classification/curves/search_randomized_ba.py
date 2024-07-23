"""Create the compression maps for the randomized compression curve."""

import logging
from datetime import datetime
import os
import glob
import functools
from typing import Tuple
import numpy as np
from scipy.stats import rankdata
from datasets import Dataset

from classification.train_model_grid import Food101, DatasetData

from utils.drr_estimation_tools import (
    calculate_dataset_drr_or_size,
)
from classification.curves.search_deterministic_ba import set_up_data
from classification.resnet_model import ResNet50
from utils.expt_utils import make_logger, binary_search_largest_below_function
from utils.consts import FOOD101_PATH, FOOD101_SS


def get_subset(ds: Dataset, target_s: int, suffix: str = "") -> Dataset:
    """
    Get the same image subset as used in the deterministic curve.

    Args:
        ds: Huggingface dataset
        target_s: target storage size
        suffix: str, suffix for the subset file

    Returns:
        subset: Huggingface dataset
    """
    subset_file = glob.glob(
        os.path.join(FOOD101_PATH, f"deterministic", f"idxs_s{target_s}_n*{suffix}.npy")
    )
    if len(subset_file) != 1:
        raise ValueError(f"Expected 1 subset file, found {len(subset_file)}")
    subset_idxs = np.load(subset_file[0])
    logging.info(f"Loaded subset from {subset_file[0]}")
    subset_idxs = set(subset_idxs)
    subset = ds.filter(lambda example: example["idx"] in subset_idxs)
    subset = subset.rename_column("idx", "original_idx")
    subset = subset.add_column("idx", range(len(subset)))
    logging.info(f"Subset size: {len(subset):,} images")
    return subset


def get_comp_levels(
    min_compression: float,
    max_compression: float,
    fake_score: np.ndarray,
    dataset_data: DatasetData,
) -> np.ndarray:
    """
    Get the compression levels for the randomized compression curve based on parameters.

    Args:
        min_compression: float, minimum compression level
        max_compression: float, maximum compression level
        fake_score: np.ndarray, random scores used for shuffling the images
        dataset_data: DatasetData describing the dataset

    Returns:
        cls: np.ndarray, compression levels
    """
    cls = (1 - fake_score / np.max(fake_score)) * (
        max_compression - min_compression
    ) + min_compression
    cls[cls < dataset_data.zero_drr_estimator()] = 0
    return cls


def get_compression_fn(
    subset: Dataset,
    target_s: int,
    min_compression: float,
    max_compression: float,
    dataset_data: DatasetData,
    fake_score: np.ndarray,
) -> Tuple[callable, str]:
    """
    Get the compression function and normalization for the randomized compression curve.

    Move the maximum compression level to achieve the target storage size. If the maximum compression level
    is too high, move the minimum compression level to achieve the target storage size.

    Args:
        subset: Huggingface dataset
        target_s: target storage size
        min_compression: float, minimum compression level
        max_compression: float, maximum compression level
        dataset_data: DatasetData
        fake_score: np.ndarray, random scores for the images

    Returns:
        compression_fn: callable, function to get the compression levels
        normalization: str, normalization
    """
    max_max = get_comp_levels(
        min_compression, max_compression, fake_score, dataset_data
    )
    max_max_size = calculate_dataset_drr_or_size(
        subset,
        max_max,
        from_img_path=False,
        image_col=dataset_data.image_col,
        size=True,
        from_preloaded=False,
    )
    if max_max_size > target_s:
        logging.info(f"Using min compression: {max_max_size} for {target_s}")
        compression_fn = functools.partial(
            get_comp_levels,
            max_compression=max_compression,
            fake_score=fake_score,
            dataset_data=dataset_data,
        )
        normalization = "min"
    else:
        logging.info(f"Using max compression: {max_max_size} for {target_s}")
        compression_fn = functools.partial(
            get_comp_levels,
            min_compression=min_compression,
            fake_score=fake_score,
            dataset_data=dataset_data,
        )
        normalization = "max"
    return compression_fn, normalization


def get_ds_size_fn(
    neg_dist: float,
    compression_fn: callable,
    dataset_data: DatasetData,
    subset: Dataset,
    normalization: str,
) -> int:
    """
    Get the dataset size function for the binary search.

    Args:
        neg_dist: float, negative distance, using negative dist to make size monotonically increasing for binary search
        compression_fn: callable, function to get the compression levels
        dataset_data: DatasetData describing the dataset
        subset: Huggingface dataset
        normalization: str, "min" or "max"

    Returns:
        int, dataset size in bytes
    """
    # using negative dist to make size monotonically increasing for binary search
    args = {f"{normalization}_compression": -neg_dist}
    compression = compression_fn(**args)
    return calculate_dataset_drr_or_size(
        subset,
        compression,
        from_img_path=False,
        image_col=dataset_data.image_col,
        size=True,
        from_preloaded=False,
    )


def get_randomized_compression_map() -> None:
    """
    Get the compression maps for the randomized compression curve.

    Saves a map from image paths to compression levels for each target storage size.
    """
    log_file = os.path.join(
        FOOD101_PATH,
        "randomized",
        f"compression_levels_{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    make_logger(log_file)

    dataset_data = Food101()
    model = ResNet50(dataset_data, hyperparameters={})
    ds = set_up_data(model)

    logging.info(f"Target ss: {FOOD101_SS}")
    min_compression = 0.5
    max_compression = 15
    logging.info(f"Min compression: {min_compression}")
    logging.info(f"Max compression: {max_compression}")

    for seed in [42, 0, 1]:
        suffix = f"_seed{seed}" if seed != 42 else ""
        for i, target_s in enumerate(FOOD101_SS):
            save_path = os.path.join(
                FOOD101_SS,
                "randomized",
                f"shuffle_{target_s}_min{min_compression}_max{max_compression}{suffix}_compression_levels.npy",
            )
            if os.path.exists(save_path):
                logging.info(f"Skipping {target_s}, already exists")
                continue
            logging.info(f"Target s: {target_s:,}")
            subset = get_subset(ds, target_s, suffix=suffix)

            np.random.seed(2024)
            fake_score = rankdata(np.random.rand(len(subset)))

            compression_fn, normalization = get_compression_fn(
                subset,
                target_s,
                min_compression,
                max_compression,
                dataset_data,
                fake_score,
            )

            ds_size_fn = functools.partial(
                get_ds_size_fn,
                compression_fn=compression_fn,
                dataset_data=dataset_data,
                subset=subset,
                normalization=normalization,
            )

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

            compression_level = compression_fn(
                **{f"{normalization}_compression": -opt_minmax}
            )
            logging.info(f"Saving to {save_path}")
            np.save(save_path, compression_level)


if __name__ == "__main__":
    get_randomized_compression_map()
