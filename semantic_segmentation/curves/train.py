"""Train the deterministic, uncompressed, and randomized curves."""

from utils.expt_utils import set_cuda_device

set_cuda_device()
import os
from typing import List
import re
import random
import torch
import numpy as np
import glob
from mmengine.config import Config
from mmengine.logging import print_log
from subprocess import Popen, PIPE
import simplejson as json

from semantic_segmentation.train_model_grid import get_train_script
from classification.curves.train_deterministic_curve import extract_n_optdist
from utils.consts import CITYSCAPES_BASE_CONFIG, CITYSCAPES_PATH, CITYSCAPES_SS


def update_config(
    cfg: Config,
    dist: float,
    subset_file: str,
    random_seed: int,
    workdir_suffix: str,
    dist_map: dict = None,
):
    """
    Update the config for training a model with a given distance and subset file.

    Args:
        cfg (Config): The config to update.
        dist (float): The Butteraugli distance.
        subset_file (str): The path to the subset file.
        random_seed (int): The random seed.
        workdir_suffix (str): The suffix for the work directory.
        dist_map (dict): The distance mapping.
    """
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="JXLTransform", distance=dist, distance_map=dist_map),
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

    cfg.work_dir = os.path.join(cfg.work_dir, workdir_suffix)
    print(f"work_dir: {cfg.work_dir}")

    cfg.train_dataloader.dataset.ann_file = subset_file

    cfg.randomness = dict(seed=random_seed)


def train(
    dist: float,
    subset_file: str,
    workdir_suffix: str,
    random_seed: int,
    resume: bool = False,
    dist_map: dict = None,
):
    """
    Train a model with a given distance and subset file.

    Args:
        dist (float): The Butteraugli distance.
        subset_file (str): The path to the subset file.
        workdir_suffix (str): The suffix for the work directory.
        random_seed (int): The random seed.
        resume (bool): Whether to resume training.
        dist_map (dict): The distance mapping.
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
    original_config = os.path.join(CITYSCAPES_BASE_CONFIG)

    cfg = Config.fromfile(original_config)
    update_config(
        cfg, dist, subset_file, random_seed, workdir_suffix, dist_map=dist_map
    )
    mmdet_train_config = original_config[:-3] + "-tmp.py"
    cfg.dump(mmdet_train_config)

    # Train the model
    gpus = torch.cuda.device_count()
    mmseg_train_script = get_train_script(gpus)
    cmd = [
        "bash" if gpus > 1 else "python",
        mmseg_train_script,
        mmdet_train_config,
    ]
    if gpus > 1:
        cmd.append(str(gpus))

    if resume:
        cmd.append("--resume")
        print("RESUMING TRAINING")

    cmd_str = " ".join(cmd)
    print(cmd_str)
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8").strip())
    print(stderr.decode("utf-8").strip())


def train_deterministic_curve(
    skip_s: List[int] = None, resume: bool = False, random_seed: int = None
):
    """
    Train the deterministic curve.

    Args:
        skip_s (List[int]): The storage sizes to skip.
        resume (bool): Whether to resume training.
        random_seed (int): The random seed for reproducibility.
    """
    suffix = f"_seed{random_seed}" if random_seed is not None else ""
    det_dists_log = os.path.join(
        CITYSCAPES_PATH, "deterministic", f"opt_compression_dists{suffix}.log"
    )
    ss, ns, dists = extract_n_optdist(det_dists_log)
    for s, n, dist in zip(ss, ns, dists):
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}, n={n}, dist={dist}")
            continue
        print(f"\ns={s} n={n} dist={dist}")
        deterministic_subset_file = os.path.join(
            CITYSCAPES_PATH, "deterministic", f"idxs_s{s}_n{n}{suffix}.npy"
        )
        print(f"deterministic_subset_file: {deterministic_subset_file}")
        print(os.path.exists(deterministic_subset_file))
        workdir_suffix = f"deterministic/s{s}{suffix}"
        if random_seed is None:
            random_seed = n
        train(
            dist,
            deterministic_subset_file,
            workdir_suffix,
            random_seed=random_seed,
            resume=resume,
        )


def train_uncompressed_curve(
    skip_s: List[int] = None, random_seed: int = None, resume: bool = False
):
    """
    Train the uncompressed curve.

    Args:
        skip_s (List[int]): The storage sizes to skip.
        random_seed (int): The random seed for reproducibility.
        resume (bool): Whether to resume training.
    """
    for i, s in enumerate(CITYSCAPES_SS):
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}")
            continue
        print(f"\ns={s}")
        suffix = f"_seed{random_seed}" if random_seed is not None else ""
        uncompressed_subset_file = os.path.join(
            CITYSCAPES_PATH, "uncompressed", f"uncompressed_s{s}{suffix}.txt"
        )
        print(f"uncompressed_subset_file: {uncompressed_subset_file}")
        print(os.path.exists(uncompressed_subset_file))
        workdir_suffix = f"uncompressed/s{s}{suffix}"
        if os.path.exists(
            os.path.join(CITYSCAPES_PATH, workdir_suffix, "iter_80000.pth")
        ):
            print(f"Skipping {workdir_suffix}, already trained")
            continue
        if random_seed is None:
            random_seed = s
        train(
            dist=0,
            subset_file=uncompressed_subset_file,
            workdir_suffix=workdir_suffix,
            random_seed=random_seed,
            resume=resume,
        )


def train_randomized_curve(
    skip_s: List[int] = None,
    resume: bool = False,
    min_compression: float = 0,
    max_compression: float = 15,
    random_state: int = None,
):
    """
    Train the randomized curve.

    Args:
        skip_s (List[int]): The storage sizes to skip.
        resume (bool): Whether to resume training.
        min_compression (float): The minimum Butteraugli distance.
        max_compression (float): The maximum Butteraugli distance.
        random_state (int): The random seed for reproducibility.
    """
    print("Training randomized curve")
    for s in CITYSCAPES_SS:
        if skip_s is not None and s in skip_s:
            print(f"Skipping s={s}")
            continue
        print(f"\n========== s={s} ==========")
        suffix = f"_seed{random_state}" if random_state is not None else ""
        compression_levels_path = os.path.join(
            CITYSCAPES_PATH,
            "randomized",
            f"shuffle_{s}_min{min_compression}_max{max_compression}{suffix}_compression_levels.json",
        )
        print(f"compression_levels_path: {compression_levels_path}")
        with open(compression_levels_path, "r") as file:
            dist_map = json.load(file)

        deterministic_subset_file = glob.glob(
            os.path.join(
                CITYSCAPES_PATH, "deterministic", f"idxs_s{s}_n[0-9]*{suffix}.npy"
            )
        )
        print(deterministic_subset_file)
        assert len(deterministic_subset_file) == 1
        deterministic_subset_file = deterministic_subset_file[0]
        print(f"deterministic_subset_file: {deterministic_subset_file}")
        print(os.path.exists(deterministic_subset_file))
        suffix += f"_max{max_compression}" if max_compression != 15 else ""
        workdir_suffix = f"randomized/s{s}{suffix}"
        if os.path.exists(
            os.path.join(CITYSCAPES_PATH, workdir_suffix, "iter_80000.pth")
        ):
            print(f"Skipping {workdir_suffix}, already trained")
            continue
        if random_state is None:
            match = re.search(r"_n(\d+)", deterministic_subset_file.split("/"))
            random_state = int(match.group(1))
        train(
            dist=0,
            dist_map=dist_map,
            subset_file=deterministic_subset_file,
            workdir_suffix=workdir_suffix,
            random_seed=random_state,
            resume=resume,
        )


if __name__ == "__main__":
    train_deterministic_curve(random_seed=42)
    train_deterministic_curve(random_seed=0)
    train_deterministic_curve(random_seed=1)

    train_uncompressed_curve(random_seed=42)
    train_uncompressed_curve(random_seed=0)
    train_uncompressed_curve(random_seed=1)

    train_randomized_curve(random_state=42)
    train_randomized_curve(random_state=0)
    train_randomized_curve(random_state=1)
