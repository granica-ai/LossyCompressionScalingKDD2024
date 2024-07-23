"""Train uncompressed curve for Food101."""

from utils.expt_utils import set_cuda_device

set_cuda_device()

import logging

import numpy as np
from torch.utils.data import DataLoader
from classification.curves.train_deterministic_curve import set_up_logging
from classification.resnet_model import ResNet50
from classification.train_model_grid import Food101
from utils.consts import DEFAULT_FOOD101_PARAMS, FOOD101_SS


def uncompressed_curve(
    ss: list,
    model: ResNet50,
    logger: logging.Logger,
    test_loader: DataLoader,
    save_indices: bool = False,
    eval_only: bool = False,
    skip_s_values: list = None,
    random_state: int = 42,
) -> None:
    """
    Train uncompressed curve for Food101.

    Args:
        ss: list, storage sizes in bytes
        model: ResNet50
        logger: logging.Logger
        test_loader: DataLoader
        save_indices: bool, whether to save the indices
        eval_only: bool, whether to only evaluate the model
        skip_s_values: list, storage sizes to skip
        random_state: int, random state for reproducibility
    """
    logging.info("Starting training uncompressed curve")
    for s in ss:
        if skip_s_values is not None and s in skip_s_values:
            continue
        if not eval_only:
            logger.info(f"\n====== Storage size: {s:,} ======\n")
            train_loader = model.get_unprocessed_data_loader(
                "train",
                size=s,
                distance=0,
                compress_before_crop=True,
                random_state=random_state,
            )
            logger.info(
                f"Number of points in uncompressed subset: {len(train_loader.dataset)}"
            )
            if save_indices:
                idxs = np.array([x["idx"] for x in train_loader.dataset])
                np.save(model.save_path + f"idxs_{s}.npy", idxs)
                logging.info(f"Saved indices to {model.save_path + f'idxs_{s}.npy'}")
            resnet = model.train(train_loader, distance=0)
        acc = model.evaluate(resnet, test_loader)
        logger.info(f"Accuracy: {acc}")


if __name__ == "__main__":
    for i in [42, 0, 1]:
        save_suffix = "uncompressed"
        if i == 42:
            save_name_suffix = ""
        else:
            save_name_suffix = f"_seed{i}"
        data = Food101()
        logger = set_up_logging(data.name, save_suffix)
        logger.info(f"Hyperparameters: {DEFAULT_FOOD101_PARAMS}")
        logger.info(f"Storage sizes: {FOOD101_SS}")

        model = ResNet50(
            data,
            hyperparameters=DEFAULT_FOOD101_PARAMS,
            save_suffix=save_suffix,
            save_name_suffix=save_name_suffix,
        )
        test_loader = model.get_unprocessed_data_loader("test")
        uncompressed_curve(
            FOOD101_SS,
            model,
            logger,
            test_loader,
            save_indices=True,
            random_state=i,
        )
