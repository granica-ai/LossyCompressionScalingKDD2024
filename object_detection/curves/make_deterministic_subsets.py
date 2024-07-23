"""Find the deterministic compression level to achieve a given storage size for the iSAID dataset."""

import os
import logging
from typing import List, Tuple
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool
import functools
from tqdm import tqdm
import simplejson as json

from utils.deteriministic_curve_utils import (
    estimate_opt_n_l,
    get_dist2l,
    plot_compression_levels,
)
from utils.consts import (
    ISAID_ANN_PATH,
    ISAID_DATA_ROOT,
    ISAID_PATH,
    ISAID_SIZES,
    ISAID_SS,
)
from utils.drr_estimation_tools import image_size_at_dist
from utils.expt_utils import binary_search_largest_below_function, make_logger


def get_fun_for_fixed_s(s: int) -> callable:
    """
    Get the scaling curve function for a fixed s.

    Args:
        s: int, storage size in bytes

    Returns:
        callable, function to minimize
    """
    params = [
        1.0022450366694422,
        269996.37601695274,
        0.09093488964456031,
        1.6962044400861096,
        0.3331982003692264,
    ]
    a, b, alpha, beta, c = params
    logging.info(f"Using params: {params} = [a, b, alpha, beta, c]")

    def fun(x):
        return a * (x**-alpha) + b * ((s / x) ** -beta) + c

    return fun


def fit_curve(
    opt_ls: List[float], dist2l: dict
) -> Tuple[callable, np.ndarray, List[float]]:
    """
    Fit a curve to the Butteraugli distance vs L data.

    Finds initialization parameters by fitting a linear function to 1/L.

    Args:
        opt_ls: List[float], optimal L values
        dist2l: dict, mapping from Butteraugli distance to L

    Returns:
        Tuple[callable, np.ndarray, List[float]], curve function, optimal parameters, estimated distances
    """
    dists = np.array(list(dist2l.keys()))
    pixels = np.array(list(dist2l.values()))
    print("dists: ", dists)
    print("pixels: ", pixels)

    # fit curve to reciprocal to get starting value
    pre_curve = lambda x, a, b: a * x + b
    p0, _ = curve_fit(pre_curve, dists, [1 / x for x in pixels])
    print("p0: ", p0)

    curve = lambda x, a, b: 1 / (a * x + b)
    popt, pcov, _, msg, _ = curve_fit(curve, dists, pixels, full_output=True, p0=p0)
    print("msg: ", msg)
    print("popt: ", popt)
    curve_inv = lambda y, a, b: (1 - y * b) / (y * a)
    estimated_dists = [curve_inv(l, *popt) for l in opt_ls]
    return curve, popt, estimated_dists


def estimate_n_and_dist(
    s_range: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Estimate optimal n and L for a range of s values and plot the results.

    Args:
        s_range: List[int], storage sizes in bytes

    Returns:
        Tuple[List[float], List[float], List[float]], optimal n and L values, estimated distances
    """
    logging.info(f"Using s_range: {s_range}")
    opt_ns, opt_ls = estimate_opt_n_l(s_range, get_fun_for_fixed_s)
    dist2l = get_dist2l(ISAID_SIZES)
    curve, popt, estimated_dists = fit_curve(opt_ls, dist2l)

    for s, n, l, dist in zip(s_range, opt_ns, opt_ls, estimated_dists):
        logging.info(f"{s=:,} - {n=:,.1f}, {l=:,.1f}, {dist=:,.1f}")
    plot_compression_levels(opt_ls, dist2l, opt_ns, curve, popt)
    opt_ns = [int(n) for n in opt_ns]
    return opt_ns, opt_ls, estimated_dists


def get_image_subset_indices(
    s_range: List[int], opt_ns: List[float], subset_path: str, random_state: int = 42
):
    """
    Create a subset of images and annotations from the iSAID dataset.

    Args:
        s_range: List[int], storage sizes in bytes
        opt_ns: List[float], optimal n values
        subset_path: str, path to save the subset
        random_state: int, random seed
    """
    # Load the original COCO format JSON file
    with open(ISAID_ANN_PATH, "r") as file:
        coco_data = json.load(file)

    n = len(coco_data["images"])
    img_ids = sorted([img["id"] for img in coco_data["images"]])
    # Shuffling because not sure if the images are ordered in any way (does disrupt patches though)
    img_ids = np.random.default_rng(random_state).choice(
        img_ids, size=len(img_ids), replace=False
    )

    images_so_far = []
    annotations_so_far = []

    for s, n in zip(s_range, opt_ns):
        subset_file = subset_path.format(s=s, n=int(n), random_state=random_state)
        os.makedirs(os.path.dirname(subset_file), exist_ok=True)
        logging.info(f"Creating subset with {n} images... {len(images_so_far)} to {n}.")
        # get new images and annotations
        new_ids = img_ids[len(images_so_far) : n]
        new_imgs = [img for img in coco_data["images"] if img["id"] in new_ids]
        new_annos = [
            anno for anno in coco_data["annotations"] if anno["image_id"] in new_ids
        ]
        images_so_far.extend(new_imgs)
        annotations_so_far.extend(new_annos)

        # Create a new JSON structure for the subset
        subset_data = {
            "images": images_so_far,
            "annotations": annotations_so_far,
            "categories": coco_data["categories"],
        }

        # Save the subset to a new JSON file
        with open(subset_file, "w") as outfile:
            json.dump(subset_data, outfile, indent=4)
        logging.info(f"Subset saved to {subset_file}.")


def get_dataset_size(distance: int, images: List[str], mult: int = 1) -> int:
    """
    Get the size of a dataset of images at a given compression distance.

    Args:
        distance: int, Butteraugli distance
        images: List[str], list of image filenames
        mult: int, multiplier for distance

    Returns:
        int, size of the dataset in bytes
    """
    func = functools.partial(get_image_size, distance=mult * distance)
    with Pool(10) as p:
        sizes = list(tqdm(p.imap(func, images), total=len(images)))
    logging.debug(f"Sizes: {sizes}")
    return sum(sizes)


def get_image_size(img_file: str, distance: float) -> int:
    """
    Get the size of an image after compression to a given Butteraugli distance.

    Args:
        img_file: str, image filename
        distance: float, Butteraugli distance

    Returns:
        int, size of the compressed image in bytes
    """
    img_path = os.path.join(ISAID_DATA_ROOT, "train/images", img_file)
    return image_size_at_dist(distance, img_path)


def make_deterministic_subsets(random_state=42):
    suffix = f"_seed{random_state}" if random_state != 42 else ""
    log_file = os.path.join(
        ISAID_PATH, f"deterministic/opt_compression_dists{suffix}.log"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    make_logger(log_file)

    logging.info(f"Using s_range: {ISAID_SS}")

    subset_path = os.path.join(
        ISAID_PATH, "deterministic", "idxs_s{s}_n{n}_seed{random_state}.json"
    )

    opt_ns, opt_ls, estimated_dists = estimate_n_and_dist(ISAID_SS)
    get_image_subset_indices(ISAID_SS, opt_ns, subset_path, random_state=random_state)

    initial_ranges = [(-x - 1.5, -x + 0.5) for x in estimated_dists]
    logging.info(f"Initial ranges: {initial_ranges}")
    for i, target_s in enumerate(ISAID_SS):
        logging.info(f"Target s: {target_s:,}")
        subset_file = subset_path.format(
            s=target_s, n=int(opt_ns[i]), random_state=random_state
        )
        with open(subset_file, "r") as f:
            subset = json.load(f)

        images = [example["file_name"] for example in subset["images"]]

        left, right = initial_ranges[i]

        opt_dist, resulting_size = binary_search_largest_below_function(
            functools.partial(get_dataset_size, images=images, mult=-1),
            target_s,
            left=left,
            right=right,
            precision=1e-3,  # * target_s,
        )
        logging.info(f"Opt dist: {opt_dist}")
        logging.info(f"Resulting size: {resulting_size}")
        logging.info(f"% Error: {(resulting_size - target_s) / target_s * 100:.2f}")


if __name__ == "__main__":
    make_deterministic_subsets()
    make_deterministic_subsets(random_state=0)
    make_deterministic_subsets(random_state=1)
