"""Find the deterministic compression level to achieve a given storage size for the Cityscapes dataset."""

import os
import logging
from typing import List, Tuple
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import functools
from tqdm import tqdm
from multiprocessing import Pool

from utils.deteriministic_curve_utils import (
    estimate_opt_n_l,
    get_dist2l,
    plot_compression_levels,
)
from utils.drr_estimation_tools import image_size_at_dist
from utils.expt_utils import binary_search_largest_below_function, make_logger
from utils.consts import (
    CITYSCAPES_DATA_ROOT,
    CITYSCAPES_PATH,
    CITYSCAPES_SIZES,
    CITYSCAPES_SS,
)


def get_fun_for_fixed_s(s: int) -> callable:
    """
    Get the scaling curve function for a fixed s.

    Args:
        s: int, storage size in bytes

    Returns:
        callable, function to minimize
    """
    params = [
        4.286691211158588,
        428277.4784308216,
        0.58938638522845,
        1.6064150537051824,
        0.13548027044911312,
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
    dist2l = get_dist2l(CITYSCAPES_SIZES)
    curve, popt, estimated_dists = fit_curve(opt_ls, dist2l)

    for s, n, l, dist in zip(s_range, opt_ns, opt_ls, estimated_dists):
        logging.info(f"{s=:,} - {n=:,.1f}, {l=:,.1f}, {dist=:,.1f}")
    plot_compression_levels(opt_ls, dist2l, opt_ns, curve, popt)
    return opt_ns, opt_ls, estimated_dists


def get_cityscapes_paths() -> List[str]:
    """
    Get the paths to the Cityscapes images.

    Returns:
        List[str], list of image paths
    """
    subset_file = os.path.join(CITYSCAPES_PATH, "subsets", "train1.0.txt")
    with open(subset_file, "r") as f:
        ds = f.readlines()
    ds = [x.strip() for x in ds]
    return ds


def get_image_subset_indices(
    s_range: List[int],
    opt_ns: List[float],
    subset_path: str,
    ds_paths: List[str],
    random_state: int = 42,
):
    """
    Create a subset of images at a given n from the Cityscapes dataset.

    Args:
        s_range: List[int], storage sizes in bytes
        opt_ns: List[float], optimal n values
        subset_path: str, path to save the subset
        ds_paths: List[str], list of image paths
        random_state: int, random seed
    """
    for s, n in zip(s_range, opt_ns):
        subset_file = subset_path.format(s=s, n=int(n), random_state=random_state)
        if os.path.exists(subset_file):
            logging.info(f"Skipping create subset for {s=:,}, {n=:,}, already exists")
            continue
        subset = train_test_split(
            ds_paths, train_size=int(n), random_state=random_state
        )[0]
        with open(subset_file, "w") as f:
            f.write("\n".join(subset))
        logging.info(
            f"Created subset for {s=:,}, {n=:,}, saved subset_idxs to {subset_file}"
        )


def get_dataset_size(distance: float, images: List[str], mult: int = 1) -> int:
    """
    Get the size of a dataset of images at a given compression distance.

    Args:
        distance: float, Butteraugli distance
        images: List[str], list of image filenames
        mult: int, multiplier for distance

    Returns:
        int, size of the dataset in bytes
    """
    func = functools.partial(image_size_at_dist, distance=mult * distance)
    with Pool(10) as p:
        sizes = list(tqdm(p.imap(func, images), total=len(images)))
    logging.debug(f"Sizes: {sizes}")
    return sum(sizes)


def make_deterministic_subsets(random_state: int = 42):
    """
    Make deterministic subsets of the Cityscapes dataset at each storage size.

    Args:
        random_state: int, random seed
    """
    suffix = f"_seed{random_state}" if random_state != 42 else ""
    log_file = os.path.join(
        CITYSCAPES_PATH, "deterministic", f"opt_compression_dists{suffix}.log"
    )
    make_logger(log_file)

    logging.info(f"Using s_range: {CITYSCAPES_SS}")

    data_root = os.path.join(CITYSCAPES_DATA_ROOT, "train")
    data_suffix = "_leftImg8bit.png"
    subset_path = os.path.join(
        CITYSCAPES_PATH, "deterministic", "idxs_s{s}_n{n}_seed{random_state}.npy"
    )

    opt_ns, opt_ls, estimated_dists = estimate_n_and_dist(CITYSCAPES_SS)
    ds = get_cityscapes_paths()
    get_image_subset_indices(
        CITYSCAPES_SS, opt_ns, subset_path, ds, random_state=random_state
    )

    initial_ranges = [(-x - 1.5, -x + 0.5) for x in estimated_dists]
    logging.info(f"Initial ranges: {initial_ranges}")
    for i, target_s in enumerate(CITYSCAPES_SS):
        logging.info(f"Target s: {target_s:,}")
        subset_file = subset_path.format(
            s=target_s, n=int(opt_ns[i]), random_state=random_state
        )
        with open(subset_file, "r") as f:
            subset = f.readlines()
        subset = [os.path.join(data_root, x.strip() + data_suffix) for x in subset]

        left, right = initial_ranges[i]

        opt_dist, resulting_size = binary_search_largest_below_function(
            functools.partial(get_dataset_size, images=subset, mult=-1),
            target_s,
            left=left,
            right=right,
            precision=1e-3,
        )
        logging.info(f"Opt dist: {opt_dist}")
        logging.info(f"Resulting size: {resulting_size}")
        logging.info(f"% Error: {(resulting_size - target_s) / target_s * 100:.2f}")


if __name__ == "__main__":
    make_deterministic_subsets()
    make_deterministic_subsets(random_state=0)
    make_deterministic_subsets(random_state=1)
