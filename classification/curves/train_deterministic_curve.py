"""Train the deterministic curve models for Food101 dataset."""

from utils.expt_utils import make_logger, set_cuda_device

set_cuda_device()

from datetime import datetime
import re
import logging
from typing import Tuple, List

import os

import numpy as np
from torch.utils.data import DataLoader
from classification.resnet_model import ResNet50
from classification.train_model_grid import Food101
from utils.consts import DEFAULT_FOOD101_PARAMS, FOOD101_PATH, FOOD101_SS


def set_up_logging(save_suffix: str, log_suffix: str = "") -> logging.Logger:
    """
    Set up logger for the experiment.

    Args:
        save_suffix: str, suffix for the save path
        log_suffix: str, suffix for the log file

    Returns:
        logging.Logger
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(FOOD101_PATH, save_suffix, f"{timestamp}{log_suffix}.log")
    return make_logger(log_path)


def extract_n_optdist(log_file_path: str) -> Tuple[list, list]:
    """
    Get optimal n and Butteraugli distance by parsing logs.

    Args:
        log_file_path: Path to the log file output of `search_deterministic_ba.py`

    Returns:
        Tuple[list, list], n values and optimal distances
    """
    n_pattern = re.compile(r"idxs_s\d+_n(\d+).npy")
    optdist_pattern = re.compile(r"Opt dist: ([-\d.]+)")

    n_values = []
    optdist_values = []

    with open(log_file_path, "r") as file:
        for line in file:
            n_match = n_pattern.search(line)
            if n_match:
                n_values.append(int(n_match.group(1)))

            optdist_match = optdist_pattern.search(line)
            if optdist_match:
                optdist_values.append(float(optdist_match.group(1)))

    return n_values, optdist_values


def get_opt_l_n(log_file_path: str = None) -> List[Tuple[int, float]]:
    if log_file_path is None:
        log_file_path = os.path.join(
            FOOD101_PATH, "deterministic", "opt_compression_dists.log"
        )
    n_values, optdist_values = extract_n_optdist(log_file_path)
    return list(zip(n_values, [-x for x in optdist_values]))


def deterministic_curve(
    ss: list,
    model: ResNet50,
    logger: logging.Logger,
    test_loader: DataLoader,
    save_indices: bool = False,
    save_name_suffix: str = "",
    random_state: int = 42,
):
    """
    Train the model on deterministic curve.

    Args:
        ss: list, storage sizes
        model: ResNet50, model to train
        logger: logging.Logger, logger for the experiment
        test_loader: DataLoader, test data loader
        save_indices: bool, whether to save the indices of the subset
        save_name_suffix: str, suffix for the save name
        random_state: int, random state for reproducibility
    """
    logging.info("Starting training deterministic curve")
    opt_l_n = get_opt_l_n()
    logging.info(f"opt_l_n: {opt_l_n}")
    for i, (n, dist) in enumerate(opt_l_n):
        logger.info(
            f"\n====== s: {ss[i]}, Number of points: {n}, Distance: {dist} ======\n"
        )
        train_loader = model.get_unprocessed_data_loader(
            "train",
            num_samples=n,
            distance=dist,
            add_idx=True,
            compress_before_crop=True,
            random_state=random_state,
        )
        if save_indices:
            idxs = np.array([x["idx"] for x in train_loader.dataset])
            save_path = model.save_path + f"idxs_{ss[i]}{save_name_suffix}.npy"
            np.save(save_path, idxs)
            logging.info(f"Saved indices to {save_path}")
        resnet = model.train(train_loader, distance=dist)

        acc = model.evaluate(resnet, test_loader)
        logger.info(f"Accuracy: {acc}")


if __name__ == "__main__":
    for i in range(2):
        save_suffix = "deterministic"
        save_name_suffix = f"_seed{i}"
        data = Food101()
        logger = set_up_logging(data.name, save_suffix)
        logger.info(f"Hyperparameters: {DEFAULT_FOOD101_PARAMS}")

        model = ResNet50(
            data,
            hyperparameters=DEFAULT_FOOD101_PARAMS,
            save_suffix=save_suffix,
            save_name_suffix=save_name_suffix,
        )
        test_loader = model.get_unprocessed_data_loader("test")

        logging.info(f"Storage sizes: {FOOD101_SS}")
        deterministic_curve(
            FOOD101_SS,
            model,
            logger,
            test_loader,
            save_name_suffix=save_name_suffix,
            random_state=i,
        )
