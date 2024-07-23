"""Train models for the randomized curve on the Food101 dataset."""

from utils.expt_utils import make_logger, set_cuda_device

set_cuda_device()

from datetime import datetime
import os
import re
from typing import List

import numpy as np
import glob
from classification.resnet_model import ResNet50
from classification.train_model_grid import Food101
from utils.consts import DEFAULT_FOOD101_PARAMS, FOOD101_PATH


def get_n(line: str) -> int:
    """
    Get the number of samples from the file name.

    Args:
        line: str, file name

    Returns:
        int, number of samples
    """
    match = re.search(r"_n(\d+)", line.split("/")[-1])
    return int(match.group(1))


def get_s(line: str) -> int:
    """
    Get the storage size from the file name.

    Args:
        line: str, file name

    Returns:
        int, storage size
    """
    match = re.search(r"_s(\d+)", line.split("/")[-1])
    return int(match.group(1))


def train_models(index_files: List[str], compression_path: str, seed: int) -> None:
    """
    Train models for the randomized curve.

    Args:
        index_files: List[str], paths to the index files
        compression_path: str, path to the compression levels
        seed: int, random seed
    """
    ns = np.array([get_n(x) for x in index_files])
    ss = np.array([get_s(x) for x in index_files])
    sort_idxs = np.argsort(ss)
    ns = ns[sort_idxs]
    ss = ss[sort_idxs]
    index_files = np.array(index_files)[sort_idxs]

    for i, n, s, path in zip(range(len(ns)), ns, ss, index_files):
        logger.info(f"\n====== Storage size: {s:,} n: {n:,} ======\n")

        compression_level_path = compression_path.format(s=s)
        logger.info(f"Loading compression levels from {compression_level_path}")
        logger.info(f"Loading subset index from {path}")

        train_loader = model.get_data_loader_file_compression(
            split="train",
            compression_file_path=compression_level_path,
            num_samples=n,
            idx_check=path,
            compress_before_crop=True,
            random_state=seed,
        )
        resnet = model.train(train_loader, distance=0)
        acc = model.evaluate(resnet, test_loader)
        logger.info(f"Accuracy: {acc}")
        print(f"Accuracy: {acc}")


if __name__ == "__main__":
    save_suffix = "randomized_max15"
    for seed in [None, 0, 1]:
        suffix = f"_seed{seed}" if seed is not None else ""
        data = Food101()

        timestamp = datetime.now().strftime("%Y-%m-%d")
        model_path = os.path.join(FOOD101_PATH, save_suffix)
        os.makedirs(model_path, exist_ok=True)

        log_file = os.path.join(model_path, f"{timestamp}_max15{suffix}.log")
        logger = make_logger(log_file)

        model = ResNet50(
            data,
            hyperparameters=DEFAULT_FOOD101_PARAMS,
            save_suffix=save_suffix,
            save_name_suffix=suffix,
        )
        test_loader = model.get_unprocessed_data_loader("test")
        compression_path = os.path.join(
            FOOD101_PATH,
            "randomized",
            "shuffle_{s}_" + f"min0.5_max15{suffix}_compression_levels.npy",
        )
        index_files = glob.glob(
            FOOD101_PATH, "deterministic", f"idxs_s*_n*{suffix}.npy"
        )
        logger.info(f"Found {len(index_files)} index files")
        train_models(index_files, compression_path, seed)
