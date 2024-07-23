"""Train the deterministic, uncompressed, and randomized curves."""

from utils.expt_utils import set_cuda_device

set_cuda_device()
import os


import re
import random
import torch
import numpy as np
import datetime
from typing import List, Tuple

import glob


from mmengine.config import Config
from mmengine.logging import print_log
from subprocess import Popen, PIPE
import simplejson as json

from utils.consts import (
    ISAID_DATA_ROOT,
    ISAID_ORIGINAL_CONFIG,
    ISAID_PATH,
    ISAID_SS,
    ISAID_TMP_CONFIG_PATH,
    MAX_GPUS,
    MMDET_PATH,
)


def extract_n_optdist(log_file_path: str) -> Tuple[List[int], List[int], List[float]]:
    """
    Use regex to extract s, n, and compression distance from a `make_deterministic_subsets.py`log file.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        Tuple[List[int], List[int], List[float]]: The n values, the s values, and the compression levels.
    """
    n_pattern = re.compile(r"s=([\d,]+) - n=([\d,.]+)")
    optdist_pattern = re.compile(r"Opt dist: -([\d.]+)")

    s_values = []
    n_values = []
    optdist_values = []

    with open(log_file_path, "r") as file:
        for line in file:
            n_match = n_pattern.search(line)
            if n_match:
                s_values.append(int("".join(n_match.group(1).split(","))))
                n_values.append(int(float("".join(n_match.group(2).split(",")))))

            optdist_match = optdist_pattern.search(line)
            if optdist_match:
                optdist_values.append(float(optdist_match.group(1)))

    return s_values, n_values, optdist_values


def update_config(
    cfg: Config,
    dist: float,
    subset_file: str,
    random_seed: int,
    workdir_suffix: str,
    dist_map: dict = None,
    train_root: str = None,
):
    """
    Update the config for training with JXLTransform.

    Args:
        cfg (Config): The original config.
        dist (float): The Butteraugli distance to use for JXLTransform.
        subset_file (str): The path to the subset file.
        random_seed (int): The random seed to use.
        workdir_suffix (str): The suffix to add to the work directory.
        dist_map (dict): Optional, a dictionary mapping image file paths to compression distances.
        train_root (str): Optional, an override root directory for the training data.
    """
    # update data root
    cfg.data_root = ISAID_DATA_ROOT
    if train_root is not None:
        cfg.train_dataloader.dataset.data_root = train_root
    else:
        cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.data_root = cfg.data_root

    # load JXLTransform
    cfg.custom_imports = dict(
        imports=["utils.mmdet_jxl_transform"],
        allow_failed_imports=False,
    )
    # insert JXLTransform immediately after loading image
    t0 = datetime.datetime.now()
    if dist_map is not None:
        cfg.train_pipeline.insert(
            1,
            dict(
                type="JXLTransform",
                distance="",
                distance_map=dist_map,
            ),
        )
    elif dist is not None:
        cfg.train_pipeline.insert(1, dict(type="JXLTransform", distance=dist))
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    print(f"Updating pipeline took {datetime.datetime.now() - t0}")

    # set work directory
    cfg.work_dir = os.path.join(ISAID_PATH, workdir_suffix)
    print(f"work_dir: {cfg.work_dir}")

    # update subset file
    cfg.train_dataloader.num_workers = 8
    cfg.val_dataloader.num_workers = 8
    cfg.train_dataloader.dataset.ann_file = subset_file
    cfg.val_dataloader.dataset.ann_file = os.path.join(
        ISAID_DATA_ROOT, "val/instancesonly_filtered_val.json"
    )
    cfg.test_dataloader.dataset.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.val_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file
    cfg.test_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file

    # only run evaluation with the val set at the very end (it's slow)
    cfg.train_cfg.val_interval = 12
    cfg.default_hooks.checkpoint = dict(interval=6, type="CheckpointHook")

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


def train(
    subset_file: str,
    workdir_suffix: str,
    random_seed: int,
    resume: bool = False,
    dist_map: dict = None,
    dist: float = None,
    train_root: str = None,
):
    """
    Train a model on a subset of data with JXLTransform.

    Args:
        subset_file (str): The path to the subset file.
        workdir_suffix (str): The suffix to add to the work directory.
        random_seed (int): The random seed to use.
        resume (bool): Whether to resume training.
        dist_map (dict): Optional, a dictionary mapping image file paths to compression distances.
        dist (float): Optional, the Butteraugli distance to use for JXLTransform.
        train_root (str): Optional, an override root directory for the training data.
    """
    print_log(
        f"============= Training distance {dist} {workdir_suffix} =============",
        logger="current",
    )

    # set random seed
    print_log(f"random_seed: {random_seed}", logger="current")
    np.random.seed(random_seed)
    random.seed(random_seed)

    # update config
    t0 = datetime.datetime.now()
    cfg = Config.fromfile(ISAID_ORIGINAL_CONFIG)
    update_config(
        cfg, dist, subset_file, random_seed, workdir_suffix, dist_map, train_root
    )
    mmdet_train_config = ISAID_TMP_CONFIG_PATH
    cfg.dump(mmdet_train_config)
    print(f"Updating config took {datetime.datetime.now() - t0}")

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

    if resume:
        print("RESUMING TRAINING")
        cmd.append("--resume")
    cmd_str = " ".join(cmd)
    print(cmd_str)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


def train_deterministic_curve(
    skip_s: List[int] = None, resume: bool = False, random_seed: int = 42
):
    """
    Train the deterministic curve.

    Args:
        skip_s (List[int]): Optional, a list of s values to skip.
        resume (bool): Whether to resume training, default False.
        random_seed (int): The random seed to use, default 42.
    """
    suffix = f"_seed{random_seed}" if random_seed != 42 else ""
    det_dists_log = os.path.join(
        ISAID_PATH, "deterministic", f"opt_compression_dists{suffix}.log"
    )
    ss, ns, dists = extract_n_optdist(det_dists_log)
    for s, n, dist in zip(ss, ns, dists):
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}, n={n}, dist={dist}")
            continue
        print(f"\ns={s} n={n} dist={dist}")
        deterministic_subset_file = os.path.join(
            ISAID_PATH, "deterministic", f"idxs_s{s}_n{n}_seed{random_seed}.json"
        )
        print(f"deterministic_subset_file: {deterministic_subset_file}")
        workdir_suffix = f"deterministic/s{s}{suffix}"
        if random_seed is None:
            random_seed = n
        train(
            deterministic_subset_file,
            workdir_suffix,
            random_seed=random_seed,
            resume=resume,
            dist=dist,
        )


def train_uncompressed_curve(
    skip_s: List[int] = None,
    random_seed: int = None,
    resume: bool = False,
    random_seed_add: int = 0,
):
    """
    Train the uncompressed curve.

    Args:
        skip_s (List[int]): Optional, a list of s values to skip.
        random_seed (int): The random seed to use, default None.
        resume (bool): Whether to resume training, default False.
        random_seed_add (int): The amount to add to the random seed, default 0. Used due
            to convergence issues with some random seeds.
    """
    for i, s in enumerate(ISAID_SS):
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}")
            continue
        print(f"\ns={s}")
        suffix = f"_seed{random_seed}" if random_seed != 42 else ""
        uncompressed_subset_file = os.path.join(
            ISAID_PATH, "uncompressed", f"uncompressed_s{s}{suffix}.json"
        )
        print(f"uncompressed_subset_file: {uncompressed_subset_file}")
        print(os.path.exists(uncompressed_subset_file))
        workdir_suffix = f"uncompressed/s{s}{suffix}"
        if os.path.exists(os.path.join(ISAID_PATH, workdir_suffix, "epoch_12.pth")):
            print(f"Skipping {workdir_suffix}, already trained")
            continue
        if random_seed is None:
            random_seed = s
        if random_seed_add != 0:
            print(f"random_seed_add: {random_seed_add}")
        train(
            dist=0,
            subset_file=uncompressed_subset_file,
            workdir_suffix=workdir_suffix,
            random_seed=random_seed + random_seed_add,
            resume=resume,
        )


def train_randomized_curve_at_s(
    s: int, random_state: int, min_compression: int, max_compression: int, resume: bool
):
    """
    Train a randomized curve at a specific s value.

    Args:
        s (int): The s value to train at.
        random_state (int): The random seed to use.
        min_compression (int): The minimum compression level to use.
        max_compression (int): The maximum compression level to use.
        resume (bool): Whether to resume training.
    """
    print(f"\n========== s={s} ==========")
    suffix = f"_seed{random_state}"
    # get compression levels
    compression_levels_path = os.path.join(
        ISAID_PATH,
        "randomized",
        f"shuffle_{s}_min{min_compression}_max{max_compression}{suffix}_compression_levels.json",
    )
    print(f"compression_levels_path: {compression_levels_path}")
    t0 = datetime.datetime.now()
    with open(compression_levels_path, "r") as file:
        dist_map = json.load(file)
    print(f"Loading dist_map took {datetime.datetime.now() - t0}")

    # get subset file
    deterministic_subset_file = glob.glob(
        os.path.join(ISAID_PATH, "deterministic", f"idxs_s{s}_n[0-9]*{suffix}.json")
    )
    assert len(deterministic_subset_file) == 1
    deterministic_subset_file = deterministic_subset_file[0]
    print(f"deterministic_subset_file: {deterministic_subset_file}")

    suffix += f"_max{max_compression}" if max_compression != 15 else ""
    workdir_suffix = f"randomized/s{s}{suffix}"
    if os.path.exists(os.path.join(ISAID_PATH, workdir_suffix, "epoch_12.pth")):
        print(f"Skipping {workdir_suffix}, already trained")
        return
    if random_state is None:
        match = re.search(r"_n(\d+)", deterministic_subset_file.split("/"))
        random_state = int(match.group(1))
    print(f"random_state: {random_state}")

    train(
        dist=0,
        dist_map=dist_map,
        subset_file=deterministic_subset_file,
        workdir_suffix=workdir_suffix,
        random_seed=random_state,
        resume=resume,
    )


def train_precompressed_randomized_curve_at_s(
    s: int, random_state: int, max_compression: int, resume: bool
):
    """
    Train a randomized curve at a specific s value with precompressed images.

    Args:
        s (int): The s value to train at.
        random_state (int): The random seed to use.
        max_compression (int): The maximum compression level to use.
        resume (bool): Whether to resume training.
    """
    print(f"\n========== s={s} ==========")
    suffix = f"_seed{random_state}"  # if random_state != 42 else ""

    # get subset file
    deterministic_subset_file = glob.glob(
        os.path.join(ISAID_PATH, "deterministic", f"idxs_s{s}_n[0-9]*{suffix}.json")
    )
    assert len(deterministic_subset_file) == 1
    deterministic_subset_file = deterministic_subset_file[0]

    suffix += f"_max{max_compression}" if max_compression != 15 else ""
    workdir_suffix = f"randomized/s{s}{suffix}"
    if os.path.exists(os.path.join(ISAID_PATH, workdir_suffix, "epoch_12.pth")):
        print(f"Skipping {workdir_suffix}, already trained")
        return
    if random_state is None:
        match = re.search(r"_n(\d+)", deterministic_subset_file.split("/"))
        random_state = int(match.group(1))
    print(f"random_state: {random_state}")

    train(
        dist=None,
        dist_map=None,
        subset_file=deterministic_subset_file,
        workdir_suffix=workdir_suffix,
        random_seed=random_state,
        resume=resume,
    )


def train_randomized_curve(
    skip_s: List[int] = None,
    resume: bool = False,
    min_compression: int = 0,
    max_compression: int = 15,
    random_state: int = None,
    s_seed_list: List[int] = None,
    precompressed: bool = False,
):
    """
    Train the randomized curve.

    Args:
        skip_s (List[int]): Optional, a list of s values to skip.
        resume (bool): Whether to resume training.
        min_compression (int): The minimum compression level to use.
        max_compression (int): The maximum compression level to use.
        random_state (int): The random seed to use.
        s_seed_list (List[int]): Optional, a list of s values and random seeds to train at.
        precompressed (bool): Whether to use precompressed images.
    """
    print("Training randomized curve")
    if s_seed_list is not None:
        for s, seed in s_seed_list:
            train_randomized_curve_at_s(
                s, seed, min_compression, max_compression, resume
            )
        return
    for s in ISAID_SS:
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}")
            continue
        if precompressed:
            train_precompressed_randomized_curve_at_s(
                s, random_state, max_compression, resume
            )
        else:
            train_randomized_curve_at_s(
                s, random_state, min_compression, max_compression, resume
            )


if __name__ == "__main__":
    train_uncompressed_curve(random_seed=42)
    train_uncompressed_curve(random_seed=0)
    train_uncompressed_curve(random_seed=1)

    train_randomized_curve(random_state=42)
    train_randomized_curve(random_state=0)
    train_randomized_curve(random_state=1)

    train_deterministic_curve()
    train_deterministic_curve(random_seed=0)
    train_deterministic_curve(random_seed=1)
