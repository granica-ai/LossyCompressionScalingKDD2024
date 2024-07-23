"""Train non-optimal (naive) compression curve for Food101."""

from utils.expt_utils import set_cuda_device

set_cuda_device()

import logging

import glob
import os

import numpy as np
from torch.utils.data import DataLoader
from classification.curves.train_deterministic_curve import set_up_logging
from classification.resnet_model import ResNet50
from classification.train_model_grid import Food101
from utils.consts import DEFAULT_FOOD101_PARAMS, FOOD101_PATH, FOOD101_SS


def non_opt_compressed_curve(
    ss: list,
    model: ResNet50,
    logger: logging.Logger,
    test_loader: DataLoader,
    distance: float,
    save_indices: bool = False,
    eval_only: bool = False,
    skip_s_values: list = None,
    train_models: bool = True,
    random_state: int = 42,
    required_subset: bool = False,
    idx_suffix: str = "",
) -> None:
    """
    Train non-optimal deterministic compression curve for Food101.

    Args:
        ss: list, storage sizes in bytes
        model: ResNet50
        logger: logging.Logger
        test_loader: DataLoader
        distance: float, Butteraugli distance
        save_indices: bool, whether to save the indices
        eval_only: bool, whether to only evaluate the model
        skip_s_values: list, storage sizes to skip
        train_models: bool, whether to train the models
        random_state: int, random state for reproducibility
        required_subset: bool, whether to use the required subset
        idx_suffix: str, suffix for the index file
    """
    logger.info("Starting training non-optimal deterministic compression curve")
    logger.info(f"Distance: {distance}")
    for s in ss:
        if skip_s_values is not None and s in skip_s_values:
            continue
        if not eval_only:
            if required_subset:
                paths = glob.glob(
                    os.path.join(
                        FOOD101_PATH, "deterministic", f"idxs_s{s}_n*{idx_suffix}.npy"
                    )
                )
                if idx_suffix == "":
                    paths = [x for x in paths if "seed" not in x]
                if len(paths) != 1:
                    raise ValueError(f"Expected 1 path, got {len(paths)}")
                required_subset_path = paths[0]
                logger.info(f"Loading required subset from {required_subset_path}")
            else:
                required_subset_path = None
            logger.info(f"\n====== Storage size: {s:,} ======\n")
            train_loader = model.get_unprocessed_data_loader(
                "train",
                size=s,
                distance=distance,
                compress_before_crop=True,
                random_state=random_state,
                required_subset_path=required_subset_path,
            )
            logger.info(
                f"Number of points in non-optimally compressed subset: {len(train_loader.dataset)}"
            )
            if save_indices:
                idxs = np.array(train_loader.dataset["idx"])
                save_path = model.save_path + f"idxs_{s}{model.save_name_suffix}.npy"
                np.save(save_path, idxs)
                logging.info(f"Saved indices to {save_path}")
            if train_models:
                resnet = model.train(train_loader, distance=distance)
        acc = model.evaluate(resnet, test_loader)
        logger.info(f"Accuracy: {acc}")


if __name__ == "__main__":
    save_suffix = "non_opt_compressed"
    for ba in [3, 8, 13]:
        for seed in [42, 0, 1]:
            suffix = f"_ba{ba}_subset"
            data = Food101()
            logger = set_up_logging(data.name, save_suffix)
            logger.info(f"Hyperparameters: {DEFAULT_FOOD101_PARAMS}")
            logger.info(f"Storage sizes: {FOOD101_SS}")

            model = ResNet50(
                data,
                hyperparameters=DEFAULT_FOOD101_PARAMS,
                save_suffix=save_suffix,
                save_name_suffix=suffix,
            )
            test_loader = model.get_unprocessed_data_loader("test")
            non_opt_compressed_curve(
                FOOD101_SS,
                model,
                logger,
                test_loader,
                distance=ba,
                required_subset=True,
            )
