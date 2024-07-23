"""Get the uncompressed subset for each storage size for the iSAID dataset."""

import os
import logging
import numpy as np
import datetime
import functools
from tqdm import tqdm
from multiprocessing import Pool
import simplejson as json

from object_detection.curves.make_deterministic_subsets import (
    get_image_size,
)
from utils.consts import ISAID_ANN_PATH, ISAID_PATH, ISAID_SS, ISAID_SUBSET_ROOT
from utils.expt_utils import make_logger


def set_up_logging(expt_name: str):
    """
    Set up logging for the script.

    Args:
        expt_name (str): The name of the experiment.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(ISAID_PATH, f"{expt_name}/subset_file_creation_{date}.log")
    make_logger(log_file)


def test_subset_sizes(s: int, random_seed: int):
    """
    Check the actual size of an uncompressed subset.

    Args:
        s (int): The intended size of the subset.
        random_seed (int): The random seed used to create the subset.
    """
    suffix = f"_seed{random_seed}" if random_seed != 42 else ""
    subset_file = os.path.join(ISAID_SUBSET_ROOT, f"uncompressed_s{s}{suffix}.json")
    with open(subset_file, "r") as f:
        subset = json.load(f)
    images = [example["file_name"] for example in subset["images"]]
    func = functools.partial(get_image_size, distance=0)
    with Pool(10) as p:
        sizes = list(tqdm(p.imap(func, images), total=len(images)))
    print(f"Total size: {np.sum(sizes)}")


def make_uncompressed_subset_files(random_state: int = 42):
    """
    Create uncompressed subset files for iSAID.

    Args:
        random_state (int): The random seed to use for shuffling
    """
    set_up_logging("uncompressed")
    logging.info(f"Creating uncompressed subset files for sizes {ISAID_SS}")

    with open(ISAID_ANN_PATH, "r") as file:
        coco_data = json.load(file)

    img_dict = {img["id"]: img for img in coco_data["images"]}
    img_ids = sorted(list(img_dict.keys()))
    img_ids = np.random.default_rng(random_state).choice(
        img_ids, size=len(img_ids), replace=False
    )
    logging.info(f"Shuffled image ids with seed {random_state}")

    func = functools.partial(get_image_size, distance=0)
    images = [img_dict[id]["file_name"] for id in img_ids]

    # get size of each image and calculate cumulative size at each index
    logging.info("Calculating sizes for all images")
    with Pool(10) as p:
        sizes = list(tqdm(p.imap(func, images), total=len(images)))
    cummulative_size = np.cumsum(sizes)

    for s in ISAID_SS:
        cutoff = np.searchsorted(cummulative_size, s)
        logging.info(f"Cutting off at {cutoff} images for size {s}")
        logging.info(f"Size options: {cummulative_size[cutoff-1: cutoff+1]}")
        subset_ids = img_ids[:cutoff]
        subset_imgs = [img_dict[id] for id in subset_ids]
        subset_annos = [
            anno for anno in coco_data["annotations"] if anno["image_id"] in subset_ids
        ]
        subset_data = {
            "images": subset_imgs,
            "annotations": subset_annos,
            "categories": coco_data["categories"],
        }

        suffix = f"_seed{random_state}" if random_state != 42 else ""
        subset_file = os.path.join(ISAID_SUBSET_ROOT, f"uncompressed_s{s}{suffix}.json")
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=4)
        logging.info(f"Saved subset to {subset_file}")


if __name__ == "__main__":
    make_uncompressed_subset_files()
    for seed in [0, 1]:
        make_uncompressed_subset_files(random_state=seed)
