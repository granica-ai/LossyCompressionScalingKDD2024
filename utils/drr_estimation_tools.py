"""Utility functions for DRR estimation."""

import os
import functools
import io
import os
import socket
import logging
from multiprocessing import Pool
from typing import NamedTuple, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import datasets
import PIL.Image
from tqdm import tqdm
from imagecodecs import jpegxl_encode, jpeg_encode, jpegxl_encode_jpeg, JpegxlError

NUM_WORKERS = 10 if socket.gethostname() == "aurora" else 2


def test_jpegxl_from_jpeg(sample: datasets.Dataset, image_col: str) -> bool:
    """
    Test if the image can be transcoded from JPEG to JPEG-XL.

    Args:
        sample: datasets.Dataset, the dataset sample.
        image_col: str, the column name of the image data.

    Returns:
        bool: True if the image can be transcoded, False otherwise.
    """
    try:
        jpegxl_encode_jpeg(sample[image_col]["bytes"])
        return True
    except JpegxlError as e:
        return False


def calc_sizes_from_bytes(
    sample: datasets.Dataset, level: np.ndarray, image_col: str = "image"
) -> Tuple[int, int, int]:
    """
    Calculate the size of the compressed image from the bytes data (typically decode=False).

    Args:
        sample: datasets.Dataset, the dataset sample.
        image_col: str, the column name of the image data.
        level: np.ndarray, the compression level for each image.

    Returns:
        Tuple[int, int, int]: The original size, compressed size, and lossless flag.
    """
    original_size = len(sample[image_col]["bytes"])
    dist = level[sample["idx"]]
    img = PIL.Image.open(io.BytesIO(sample[image_col]["bytes"]))
    lossless = 0
    if dist == 0:
        try:
            compressed_size = len(jpegxl_encode_jpeg(sample[image_col]["bytes"]))
        except JpegxlError as e:
            compressed_size = len(jpegxl_encode(img, lossless=True))
            print("Lossless compressed size:", compressed_size)
            lossless = 1
        return original_size, compressed_size, lossless
    compressed_size = len(jpegxl_encode(img, distance=dist))
    return original_size, compressed_size, lossless


def calc_sizes_from_path(
    sample: datasets.Dataset, level: np.ndarray, image_col: str = "pixel_values"
) -> Tuple[int, int, int]:
    """
    Calculate the size of the compressed image from the image path.

    Args:
        sample: datasets.Dataset, the dataset sample.
        image_col: str, the column name of the image data.
        level: np.ndarray, the compression level for each image.

    Returns:
        Tuple[int, int, int]: The original size, compressed size, and lossless flag.
    """
    path = sample[image_col]["path"]
    original_size = os.path.getsize(path)
    dist = level[sample["idx"]]
    if dist == 0:
        print("Warning - no lossless compression set up for calc_sizes_from_path")
        return original_size, original_size, 1
    img = PIL.Image.open(path)
    compressed_size = len(jpegxl_encode(img, distance=dist))
    return original_size, compressed_size, 0


def calc_sizes_from_preloaded_img(
    sample: datasets.Dataset, level: np.ndarray, image_col: str = "pixel_values"
) -> Tuple[int, int, int]:
    """
    Calculate the size of the compressed image from the preloaded image data.

    Args:
        sample: datasets.Dataset, the dataset sample.
        image_col: str, the column name of the image data.
        level: np.ndarray, the compression level for each image.

    Returns:
        Tuple[int, int, int]: The original size, compressed size, and lossless flag.
    """
    img = sample[image_col]
    original_size = len(jpeg_encode(img))
    dist = level[sample["idx"]]
    lossless = 0
    if dist == 0:
        compressed_size = len(jpegxl_encode(img, lossless=True))
        lossless = 1
    else:
        compressed_size = len(jpegxl_encode(img, distance=dist))
    return original_size, compressed_size, lossless


def calculate_dataset_drr_or_size(
    train: datasets.Dataset,
    level: np.ndarray,
    from_img_path: bool = False,
    image_col: str = "image",
    size: bool = False,
    from_preloaded: bool = False,
) -> float:
    """
    Calculate the dataset size or DRR.

    Args:
        train: datasets.Dataset, the dataset sample.
        level: np.ndarray, the compression level for each image.
        from_img_path: bool, whether to use the image path.
        image_col: str, the column name of the image data.
        size: bool, whether to calculate the size. If False, calculate the DRR.
        from_preloaded: bool, whether to use preloaded images.

    Returns:
        float: The dataset size or DRR.
    """
    if np.isscalar(level):
        level = np.ones(np.max(train["idx"]) + 1) * level
    if from_img_path:
        logging.info(f"Using path and image col {image_col}")
        level_calc_sizes = functools.partial(
            calc_sizes_from_path, level=level, image_col=image_col
        )
    elif from_preloaded:
        logging.info(f"Using preloaded image col {image_col}")
        level_calc_sizes = functools.partial(
            calc_sizes_from_preloaded_img, level=level, image_col=image_col
        )
    else:
        using_jpeg = test_jpegxl_from_jpeg(train[0], image_col)
        logging.info(f"Using bytes and image col {image_col}, using jpeg: {using_jpeg}")
        level_calc_sizes = functools.partial(
            calc_sizes_from_bytes, level=level, image_col=image_col
        )

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(level_calc_sizes, train), total=len(train)))

    original_size, compressed_size, lossless = zip(*results)
    logging.info(f"Num images losslessly compressed: {sum(lossless)}")
    print(f"Fraction losslessly compressed - {sum(lossless) / len(lossless)}")

    compressed_size = sum(compressed_size)
    if size:
        return compressed_size

    original_size = sum(original_size)
    drr = 1 - compressed_size / original_size
    return drr


def get_sample_of_count(
    ds: datasets.Dataset, count: int, model: NamedTuple, random_state: int = 42
) -> datasets.Dataset:
    """
    Get a sample of a given count from the dataset.

    Args:
        ds: datasets.Dataset, the dataset.
        count: int, the number of samples.
        model: NamedTuple, the model parameters.
        random_state: int, the random seed.

    Returns:
        datasets.Dataset: The subsample.
    """
    if len(ds) <= count:
        logging.warning(
            f"Dataset size {len(ds)} is smaller than count {count}, returning full dataset."
        )
        return ds
    subsample_size = int(np.round(count))
    if (
        not model.dataset_data.stratify
        or len(ds) - subsample_size < model.dataset_data.num_classes
    ):
        # don't stratify
        subsample, _ = train_test_split(
            ds, train_size=subsample_size, random_state=random_state, shuffle=True
        )
    else:
        subsample, _ = train_test_split(
            ds,
            train_size=int(np.round(count)),
            random_state=random_state,
            stratify=ds["label"],
        )
    return datasets.Dataset.from_dict(subsample)


def estimate_size_per_img(
    ds: datasets.Dataset,
    ba: float,
    model: NamedTuple,
    image_col: str,
    from_img_path: bool,
    subsample_size: int = 500,
    from_preloaded: bool = False,
) -> float:
    """
    Estimate the size per image in the dataset.

    Args:
        ds: datasets.Dataset, the dataset.
        ba: float, the Butteraugli distance to compress to .
        model: NamedTuple, the model parameters.
        image_col: str, the column name of the image data.
        from_img_path: bool, whether to use the image path.
        subsample_size: int, the size of the subsample.
        from_preloaded: bool, whether to use preloaded images.

    Returns:
        float: The size per image in bytes.
    """
    subsample = get_sample_of_count(ds, min(len(ds), subsample_size), model)
    subsample_size = calculate_dataset_drr_or_size(
        subsample,
        ba,
        image_col=image_col,
        from_img_path=from_img_path,
        size=True,
        from_preloaded=from_preloaded,
    )
    return subsample_size / len(subsample)


def get_images_for_target_size(
    current_subset: datasets.Dataset,
    target_size: int,
    ba: float,
    model: NamedTuple,
    image_col: str,
    from_img_path: bool,
    epsilon: float = 0.001,
    from_preloaded: bool = False,
    random_state: int = 42,
) -> datasets.Dataset:
    """
    Get images to reach a target size at a given compression level.

    Args:
        current_subset: datasets.Dataset, the current subset.
        target_size: int, the target size.
        ba: float, the Butteraugli distance to compress to.
        model: NamedTuple, the model parameters.
        image_col: str, the column name of the image data.
        from_img_path: bool, whether to use the image path.
        epsilon: float, the error tolerance.
        from_preloaded: bool, whether to use preloaded images.
        random_state: int, the random seed.

    Returns:
        datasets.Dataset: The subsample.
    """
    logging.info(
        f"\n====== Trying to find images for target size {target_size} at ba {ba} ======"
    )
    logging.info(f"Using random state {random_state}")
    current_error = 1
    size_per_img = estimate_size_per_img(
        current_subset,
        ba,
        model,
        image_col=image_col,
        from_img_path=from_img_path,
        from_preloaded=from_preloaded,
    )
    num_images = target_size / size_per_img
    seen_sizes = set()
    while current_error > epsilon:
        logging.info(f"Estimated num images: {num_images}")
        subsample = get_sample_of_count(
            current_subset,
            num_images,
            model,
            random_state=random_state,
        )
        if len(subsample) in seen_sizes:
            logging.warning(
                f"The search did not fully converge, size is {len(subsample)} and current error is {current_error}."
            )
            break
        seen_sizes.add(len(subsample))
        logging.info(f"Subsample count: {len(subsample)}")
        subsample_size = calculate_dataset_drr_or_size(
            subsample,
            ba,
            from_img_path=from_img_path,
            image_col=image_col,
            size=True,
            from_preloaded=from_preloaded,
        )
        logging.info(f"Subsample size: {subsample_size}")
        current_error = abs(subsample_size - target_size) / target_size
        logging.info(f"Current error: {current_error}")
        correction = (target_size - subsample_size) / (subsample_size / len(subsample))
        num_images = len(subsample) + correction

    subsample = _trim_dataset_size(
        subsample,
        subsample_size,
        target_size,
        model,
        ba,
        random_state,
        from_img_path,
        image_col,
        from_preloaded,
    )

    return subsample


def _trim_dataset_size(
    subsample: datasets.Dataset,
    subsample_size: int,
    target_size: int,
    model: NamedTuple,
    ba: float,
    random_state: int,
    from_img_path: bool,
    image_col: str,
    from_preloaded: bool,
) -> datasets.Dataset:
    """
    Trim a slightly too large dataset size to reach the target size.

    Args:
        subsample: datasets.Dataset, the subsample.
        subsample_size: int, the size of the subsample.
        target_size: int, the target size.
        model: NamedTuple, the model parameters.
        ba: float, the Butteraugli distance to compress to.
        random_state: int, the random seed.
        from_img_path: bool, whether to use the image path.
        image_col: str, the column name of the image data.
        from_preloaded: bool, whether to use preloaded images.

    Returns:
        datasets.Dataset: The trimmed subsample.
    """
    while subsample_size > target_size:
        logging.warning(
            f"Subsample size {subsample_size} is larger than target size {target_size}, "
            "removing one image"
        )
        subsample = get_sample_of_count(
            subsample,
            len(subsample) - 1,
            model,
            random_state=random_state,
        )
        subsample_size = calculate_dataset_drr_or_size(
            subsample,
            ba,
            from_img_path=from_img_path,
            image_col=image_col,
            size=True,
            from_preloaded=from_preloaded,
        )
        logging.info(
            f"New size after removing one image: {subsample_size} for {len(subsample)} images"
        )
    return subsample


def get_target_size_from_subset(
    subset: datasets.Dataset,
    extra_imgs: datasets.Dataset,
    target_size: int,
    ba: float,
    model: NamedTuple,
    image_col: str,
    from_img_path: bool,
    random_state: int = 42,
    epsilon: float = 0.001,
) -> datasets.Dataset:
    """
    Get a subset of images to reach a target size at a given compression level.

    Args:
        subset: datasets.Dataset, the current subset.
        extra_imgs: datasets.Dataset, the extra images.
        target_size: int, the target size.
        ba: float, the Butteraugli distance to compress to.
        model: NamedTuple, the model parameters.
        image_col: str, the column name of the image data.
        from_img_path: bool, whether to use the image path.
        random_state: int, the random seed.
        epsilon: float, the error tolerance.

    Returns:
        datasets.Dataset: The subsample.
    """
    fixed_size = calculate_dataset_drr_or_size(
        subset, ba, from_img_path=from_img_path, image_col=image_col, size=True
    )
    if fixed_size > target_size:
        logging.info(
            "Original subset is larger than target size, searching within subset to reach target size"
        )
        return get_images_for_target_size(
            subset,
            target_size=target_size,
            ba=ba,
            model=model,
            image_col=image_col,
            from_img_path=from_img_path,
            random_state=random_state,
        )
    else:
        logging.info(
            "Original subset is smaller than target size, searching for extra images to add to reach target"
        )
        # num_est = int(np.round((target_size - current_size) / len(subset)))
        current_error = abs(fixed_size - target_size) / target_size
        num_images = int(fixed_size / len(subset) * current_error)
        seen_sizes = set()
        while current_error > epsilon:
            # logging.info(f"\nCurrent Error: {current_error}")
            logging.info(f"Estimated num images: {num_images}")
            extras = get_sample_of_count(
                extra_imgs,
                num_images,
                model,
                random_state=random_state,
            )
            if len(extras) in seen_sizes:
                logging.warning(
                    f"The search did not fully converge, size is {len(extras)} and current error is {current_error}."
                )
                break
            seen_sizes.add(len(extras))
            logging.info(f"Subsample count: {len(extras)}")
            extras_size = calculate_dataset_drr_or_size(
                extras,
                ba,
                from_img_path=from_img_path,
                image_col=image_col,
                size=True,
                from_preloaded=False,
            )
            logging.info(f"Subsample size: {extras_size}")
            current_error = abs(extras_size + fixed_size - target_size) / target_size
            logging.info(f"Current error: {current_error}")
            correction = (target_size - extras_size - fixed_size) / (
                extras_size / len(extras)
            )
            num_images = len(extras) + correction

        extras = _trim_dataset_size(
            extras,
            extras_size,
            target_size - fixed_size,
            model,
            ba,
            random_state,
            from_img_path,
            image_col,
            from_preloaded=False,
        )

        print(subset)
        print(extras)
        extras = extras.cast(subset.features)
        print(extras)
        return datasets.concatenate_datasets(
            [
                subset.cast_column("image", datasets.Image(decode=False)),
                extras.cast_column("image", datasets.Image(decode=False)),
            ],
        )


def image_size_at_dist(img_path: str, distance: float) -> int:
    """
    Get the size of an image after compression to a given Butteraugli distance.

    Args:
        img_path: str, image filename
        distance: float, Butteraugli distance

    Returns:
        int, size of the compressed image in bytes
    """
    img = PIL.Image.open(img_path)
    img = np.array(img)
    if distance == 0:
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            num_bytes = len(jpegxl_encode_jpeg(img_bytes))
            logging.debug("Using lossless transcoding")
        except JpegxlError:
            num_bytes = len(jpegxl_encode(img, lossless=True))
            logging.debug("Using lossless encoding")
    else:
        num_bytes = len(jpegxl_encode(img, distance=distance))
        logging.debug(f"Using lossy encoding with distance {distance}")
    logging.debug(f"Size of {img_path}: {num_bytes}")
    return num_bytes
