import os
import logging
from typing import List, Tuple
import numpy as np
from scipy.optimize import curve_fit
from torchvision import transforms
import datasets


from utils.deteriministic_curve_utils import (
    estimate_opt_n_l,
    get_dist2l,
    plot_compression_levels,
)
from utils.drr_estimation_tools import (
    calculate_dataset_drr_or_size,
)
from classification.resnet_model import ResNet50
from classification.train_model_grid import Food101
from utils.consts import FOOD101_PATH, FOOD101_SIZES, FOOD101_SS
from utils.expt_utils import make_logger, binary_search_largest_below_function


def get_fun_for_fixed_s(s: int) -> callable:
    """
    Get the scaling curve function for a fixed s.

    Args:
        s: int, storage size in bytes

    Returns:
        callable, function to minimize
    """
    params = [
        6.706939196262124,
        1355.2887767756654,
        0.33330403247197804,
        1.0613126086527451,
    ]
    a, b, alpha, beta = params
    logging.info(f"Using params: {params} = [a, b, alpha, beta]")

    def fun(x):
        return a * (x**-alpha) + b * ((s / x) ** -beta)

    return fun


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
    dist2l = get_dist2l(FOOD101_SIZES)
    curve, popt, estimated_dists = fit_curve(opt_ls, dist2l)

    for s, n, l, dist in zip(s_range, opt_ns, opt_ls, estimated_dists):
        logging.info(f"{s=:,} - {n=:,.1f}, {l=:,.1f}, {dist=:,.1f}")
    plot_compression_levels(opt_ls, dist2l, opt_ns, curve, popt)
    return opt_ns, opt_ls, estimated_dists


def fit_curve(
    opt_ls: List[float], dist2l: dict
) -> Tuple[callable, np.ndarray, List[float]]:
    """
    Fit a curve to the Butteraugli distance vs L data for visualization and interpolation.

    Args:
        opt_ls: List[float], optimal L values
        food101dist2l: dict, mapping from Butteraugli distance to L

    Returns:
        Tuple[callable, np.ndarray, List[float]], curve function, parameters, estimated distances
    """
    curve = lambda x, a, b: 1 / (a * x + b)
    dists = np.array(list(dist2l.keys()))
    pixels = np.array(list(dist2l.values()))
    popt, pcov = curve_fit(curve, dists, pixels)
    curve_inv = lambda y, a, b: (1 - y * b) / (y * a)
    estimated_dists = [curve_inv(l, *popt) for l in opt_ls]
    return curve, popt, estimated_dists


def set_up_data(model: ResNet50, compress_before_crop: bool = True) -> datasets.Dataset:
    """
    Get dataset and preprocess images if needed.

    Args:
        model: ResNet50
        compress_before_crop: bool, whether to compress before cropping in preprocessing
    """
    ds = model.get_dataset(
        "train", compress_before_crop=compress_before_crop, add_idx=True
    )

    if not compress_before_crop:
        logging.info("Resizing and cropping images prior to compression")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        def _transform(example):
            try:
                example["image"] = [preprocess(x) for x in example["image"]]
            except:
                raise ValueError(f"Error in {example['image']}")
            return example

        ds = ds.map(_transform, batched=True, num_proc=4)
    ds = ds.cast_column(model.data.image_col, datasets.Image(decode=False))
    return ds


def get_image_subset_indices(
    model: ResNet50,
    s_range: List[int],
    opt_ns: List[float],
    subset_path: str,
    random_state: int = 42,
) -> None:
    """
    Make the index lists for a given n value.

    Args:
        model: ResNet50
        s_range: List[int], storage sizes in bytes
        opt_ns: List[float], optimal n values
        subset_path: str, path to save the index lists
        random_state: int, random state for reproducibility
    """
    for s, n in zip(s_range, opt_ns):
        subset_file = subset_path.format(s=s, n=int(n))
        if os.path.exists(subset_file):
            logging.info(f"Skipping create subset for {s=:,}, {n=:,}, already exists")
            continue
        subset = model.get_dataset(
            "train",
            num_samples=int(n),
            compress_before_crop=True,
            add_idx=True,
            random_state=random_state,
        )
        subset_idxs = np.array([x["idx"] for x in subset])
        os.makedirs(os.path.dirname(subset_file), exist_ok=True)
        np.save(subset_file, subset_idxs)
        logging.info(
            f"Created subset for {s=:,}, {n=:,}, saved subset_idxs to {subset_file}"
        )


def make_idx_list_and_find_opt_ba(
    suffix: str = "", skip_s_values: List[int] = None, random_state: int = 42
) -> None:
    """
    Make index lists for subsets of size n and find the optimal Butteraugli distance for a range of s values.

    Args:
        suffix: str, suffix for the log file
        skip_s_values: List[int], storage sizes to skip
        random_state: int, random state for reproducibility
    """

    log_file = os.path.join(
        FOOD101_PATH, "deterministic", f"opt_compression_dists{suffix}.log"
    )
    make_logger(log_file)

    dataset_data = Food101()
    model = ResNet50(
        dataset_data,
        hyperparameters={},
        save_suffix="deterministic",
        save_name_suffix=suffix,
    )
    if skip_s_values is None:
        skip_s_values = []

    s_range = [s for s in FOOD101_SS if s not in skip_s_values]

    subset_path = os.path.join(
        FOOD101_PATH,
        "deterministic",
        "idxs_s{s}_n{n}" + f"{suffix}.npy",
    )

    opt_ns, opt_ls, estimated_dists = estimate_n_and_dist(s_range)
    get_image_subset_indices(
        model, s_range, opt_ns, subset_path, random_state=random_state
    )
    ds = set_up_data(model, compress_before_crop=True)
    initial_ranges = [(-x - 0.1, -x + 2) for x in estimated_dists]
    logging.info(f"Initial ranges: {initial_ranges}")
    for i, target_s in enumerate(s_range):
        logging.info(f"Target s: {target_s:,}")
        subset_file = subset_path.format(s=target_s, n=int(opt_ns[i]))
        subset_idxs = np.load(subset_file)
        logging.info(f"Loaded subset_idxs from {subset_file}")
        subset_idxs = set(subset_idxs)
        subset = ds.filter(lambda example: example["idx"] in subset_idxs, num_proc=4)

        def ds_size_fn(neg_dist):
            # using negative dist to make size monotonically increasing for binary search
            return calculate_dataset_drr_or_size(
                subset,
                -neg_dist,
                from_img_path=False,
                image_col="image",
                size=True,
                from_preloaded=False,
            )

        left, right = initial_ranges[i]
        opt_dist, resulting_size = binary_search_largest_below_function(
            ds_size_fn, target_s, left=left, right=right, precision=1e-3
        )
        logging.info(f"Opt dist: {opt_dist}")
        logging.info(f"Resulting size: {resulting_size}")
        logging.info(f"% Error: {(resulting_size - target_s) / target_s * 100:.2f}")


if __name__ == "__main__":
    make_idx_list_and_find_opt_ba()
    for i in range(2):
        make_idx_list_and_find_opt_ba(suffix=f"_seed{i}", random_state=i)
