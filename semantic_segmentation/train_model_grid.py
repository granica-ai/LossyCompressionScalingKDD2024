"""Train the scaling curve fitting models for the Cityscapes dataset."""

from utils.expt_utils import set_cuda_device

set_cuda_device()
import os
import random
import torch
import numpy as np


from mmengine.config import Config
from mmengine.logging import print_log
from subprocess import Popen, PIPE

from utils.consts import (
    CITYSCAPES_BASE_CONFIG,
    CITYSCAPES_PATH,
    CITYSCAPES_SUBSET_ROOT,
    MAX_GPUS,
    MMSEG_PATH,
)


def update_config(cfg: Config, dist: float, frac: float, random_seed: int):
    """
    Update the configuration file with the experiment-specific parameters.

    Args:
        cfg (Config): The configuration file.
        dist (float): The Butteraugli distance.
        frac (float): The fraction of the dataset to use.
        random_seed (int): The random seed.
    """
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="JXLTransform", distance=dist),
        dict(type="LoadAnnotations"),
        dict(
            type="RandomResize",
            scale=(2048, 1024),
            ratio_range=(0.5, 2.0),
            keep_ratio=True,
        ),
        dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(type="PackSegInputs"),
    ]

    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

    cfg.work_dir = os.path.join(cfg.work_dir, f"dist{dist}_frac{frac}/")
    print(f"work_dir: {cfg.work_dir}")

    subset_path = os.path.join(CITYSCAPES_SUBSET_ROOT, f"train{frac:.1f}.txt")
    print(f"subset_path: {subset_path}")
    cfg.train_dataloader.dataset.ann_file = subset_path

    cfg.randomness = dict(seed=random_seed)


def get_train_script(gpus: int) -> str:
    """
    Get the training script to use based on the number of GPUs available.

    Args:
        gpus (int): The number of GPUs available.

    Returns:
        str: The path to the training script.
    """
    if gpus == 1:
        print("Using 1 GPU")
        mmseg_train_script = os.path.join(MMSEG_PATH, "tools", "train.py")
    elif gpus <= MAX_GPUS:
        print(f"Using {gpus} GPUs")
        mmseg_train_script = os.path.join(MMSEG_PATH, "tools", "dist_train.sh")
    else:
        raise ValueError(f"Unexpected number of GPUs: {gpus}")
    return mmseg_train_script


def train(dist: float, frac: float):
    """
    Train the scaling curve fitting models on the Cityscapes dataset for one grid point.

    Args:
        dist (float): The Butteraugli distance.
        frac (float): The fraction of the dataset to use
    """
    print_log(
        f"============= Training distance {dist} frac {frac} =============",
        logger="current",
    )

    # set random seed
    random_seed = int(100 * frac)
    print_log(f"random_seed: {random_seed}", logger="current")
    np.random.seed(random_seed)
    random.seed(random_seed)

    # update config
    cfg = Config.fromfile(CITYSCAPES_BASE_CONFIG)
    update_config(cfg, dist, frac, random_seed)
    mmdet_train_config = CITYSCAPES_BASE_CONFIG[:-3] + "-tmp.py"
    cfg.dump(mmdet_train_config)

    # Train the model
    gpus = torch.cuda.device_count()
    mmdet_train_script = get_train_script(gpus)
    cmd = [
        "bash" if gpus > 1 else "python",
        mmdet_train_script,
        mmdet_train_config,
    ]
    if gpus > 1:
        cmd.append(str(gpus))

    cmd_str = " ".join(cmd)
    print(cmd_str)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


def evaluate(train_dist: float, frac: float, test_dist: float = 0):
    """
    Evaluate the scaling curve fitting models on the Cityscapes dataset for one grid point.
    Used to generate the test compression grid.

    Args:
        train_dist (float): The Butteraugli distance used for training.
        frac (float): The fraction of the dataset used for training.
        test_dist (float): The Butteraugli distance used for testing.
    """
    # # Load configuration file
    cfg = Config.fromfile(CITYSCAPES_BASE_CONFIG)
    print_log(
        f"============= Testing distance {train_dist} frac {frac} at test distance {test_dist} =============",
        logger="current",
    )

    # update work_dir
    cfg.work_dir = os.path.join(
        cfg.work_dir, f"dist{train_dist}_frac{frac}/test/test_dist{test_dist}/"
    )
    print(f"work_dir: {cfg.work_dir}")

    # compress test data
    cfg.test_pipeline.insert(1, dict(type="JXLTransform", distance=test_dist))
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline

    mmseg_train_config = CITYSCAPES_BASE_CONFIG[:-3] + "-tmp.py"
    cfg.dump(mmseg_train_config)

    mmseg_test_script = os.path.join(MMSEG_PATH, "tools", "test.py")
    model_to_test = os.path.join(
        CITYSCAPES_PATH, f"dist{train_dist}_frac{frac}", "iter_80000.pth"
    )
    cmd = [
        "python",
        mmseg_test_script,
        mmseg_train_config,
        model_to_test,
    ]
    print(" ".join(cmd))
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


if __name__ == "__main__":
    # Train scaling law models
    dists = [1, 2, 4, 7, 10]
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    for dist in dists:
        for frac in fracs:
            print(f"Training distance {dist} frac {frac}")
            train(dist, frac)

    # Make test compression grid
    frac = 1.0
    dists = [1, 2, 4, 7, 10, 15]
    for train_dist in dists:
        for test_dist in dists:
            evaluate(train_dist, frac, test_dist)
