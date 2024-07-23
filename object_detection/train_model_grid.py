"""Train the scaling curve fitting models for the iSAID dataset."""

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
    ISAID_DATA_ROOT,
    ISAID_ORIGINAL_CONFIG,
    ISAID_PATH,
    ISAID_SUBSET_ROOT,
    ISAID_TMP_CONFIG_PATH,
    MMDET_PATH,
    MAX_GPUS,
)


def update_config(cfg: Config, dist: float, frac: float, random_seed: int):
    """
    Update the config for training with JXLTransform.

    Args:
        cfg (Config): The original config.
        dist (float): The Butteraugli distance to use for JXLTransform.
        frac (float): The fraction of the dataset to use.
        random_seed (int): The random seed to use.
    """
    # update data root
    cfg.data_root = ISAID_DATA_ROOT
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.data_root = cfg.data_root

    # load JXLTransform
    cfg.custom_imports = dict(
        imports=["utils.mmdet_jxl_transform"],
        allow_failed_imports=False,
    )
    # insert JXLTransform immediately after loading image
    cfg.train_pipeline.insert(1, dict(type="JXLTransform", distance=dist))
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

    # set work directory
    cfg.work_dir = os.path.join(ISAID_PATH, f"dist{dist}_frac{frac}/")
    print(f"work_dir: {cfg.work_dir}")

    # update subset file if necessary
    if frac < 1.0:
        subset_path = os.path.join(ISAID_SUBSET_ROOT, f"train{frac:.1f}.json")
    else:
        subset_path = os.path.join(
            ISAID_DATA_ROOT, "train/instancesonly_filtered_train.json"
        )
    print(f"subset_path: {subset_path}")
    cfg.train_dataloader.num_workers = 8
    cfg.val_dataloader.num_workers = 8
    cfg.train_dataloader.dataset.ann_file = subset_path
    cfg.val_dataloader.dataset.ann_file = os.path.join(
        ISAID_DATA_ROOT, "val/instancesonly_filtered_val.json"
    )
    cfg.test_dataloader.dataset.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.val_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.test_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file

    # set random seed
    cfg.randomness = dict(seed=random_seed)


def get_train_script(gpus: int) -> str:
    """
    Get the training script based on the number of GPUs.

    Args:
        gpus (int): The number of GPUs to use.

    Returns:
        str: The path to the training script.
    """

    if gpus == 1:
        print("Using 1 GPU")
        mmdet_train_script = os.path.join(MMDET_PATH, "tools/train.py")
    elif gpus <= MAX_GPUS:
        print(f"Using {gpus} GPUs")
        mmdet_train_script = os.path.join(MMDET_PATH, "tools/dist_train.sh")
    else:
        raise ValueError(f"Unexpected number of GPUs: {gpus}")
    return mmdet_train_script


def get_evaluate_script(gpus: int) -> str:
    """
    Get the evaluation script based on the number of GPUs.

    Args:
        gpus (int): The number of GPUs to use.

    Returns:
        str: The path to the evaluation script.
    """
    if gpus == 1:
        print("Using 1 GPU")
        mmdet_eval_script = os.path.join(MMDET_PATH, "tools/test.py")
    elif gpus <= MAX_GPUS:
        print(f"Using {gpus} GPUs")
        mmdet_eval_script = os.path.join(MMDET_PATH, "tools/dist_test.sh")
    else:
        raise ValueError(f"Unexpected number of GPUs: {gpus}")
    return mmdet_eval_script


def train(dist: float, frac: float):
    """
    Train the model with the given distance and fraction of the dataset.

    Args:
        dist (float): The Butteraugli distance to use for JXLTransform.
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
    cfg = Config.fromfile(ISAID_ORIGINAL_CONFIG)
    update_config(cfg, dist, frac, random_seed)
    cfg.dump(ISAID_TMP_CONFIG_PATH)

    # Train the model
    gpus = torch.cuda.device_count()
    mmdet_train_script = get_train_script(gpus)
    cmd = [
        "bash" if gpus > 1 else "python",
        mmdet_train_script,
        ISAID_TMP_CONFIG_PATH,
    ]
    if gpus > 1:
        cmd.append(str(gpus))

    print("RESUMING TRAINING")
    cmd.append("--resume")
    cmd_str = " ".join(cmd)
    print(cmd_str)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


def evaluate(dist: float, frac: float):
    """
    Evaluate the model with the given distance and fraction of the dataset.

    Args:
        dist (float): The Butteraugli distance to use for JXLTransform.
        frac (float): The fraction of the dataset to use
    """
    print_log(
        f"============= Evaluating distance {dist} frac {frac} =============",
        logger="current",
    )

    # set random seed
    random_seed = int(100 * frac)
    print_log(f"random_seed: {random_seed}", logger="current")
    np.random.seed(random_seed)
    random.seed(random_seed)

    # update config
    cfg = Config.fromfile(ISAID_ORIGINAL_CONFIG)
    update_config(cfg, dist, frac, random_seed)
    cfg.dump(ISAID_TMP_CONFIG_PATH)

    # get checkpoint
    checkpoint = os.path.join(
        cfg.work_dir,
        "epoch_12.pth",
    )

    # Train the model
    gpus = torch.cuda.device_count()
    mmdet_eval_script = get_evaluate_script(gpus)
    cmd = [
        "bash" if gpus > 1 else "python",
        mmdet_eval_script,
        ISAID_TMP_CONFIG_PATH,
        checkpoint,
    ]
    if gpus > 1:
        cmd.append(str(gpus))

    cmd_str = " ".join(cmd)
    print(cmd_str)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


if __name__ == "__main__":
    dists = [1, 2, 4, 7, 10, 15]
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    for dist in dists:
        for frac in fracs:
            print(f"Training distance {dist} frac {frac}")
            train(dist, frac)
