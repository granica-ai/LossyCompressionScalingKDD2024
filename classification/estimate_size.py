"""Estimate the size of JXL compressed images from the Food101 dataset."""

from imagecodecs import jpegxl_encode
import simplejson as json
from datasets import Dataset

from utils.consts import FOOD101_SIZES
from classification.resnet_model import load_food101_dataset


def compression_fn(sample: Dataset, dist: int) -> Dataset:
    """
    Calculate the size of the JXL compressed image.

    Args:
        sample: The dataset sample.
        dist: The distance parameter for the JXL compression.

    Returns:
        Dataset: The dataset sample with the size of the compressed image.
    """
    sample["size"] = [len(jpegxl_encode(x, distance=dist)) for x in sample["image"]]
    return sample


def calc_jxl_sizes_from_files():
    """Calculate average size of JXL compressed images from on the Food101 dataset."""
    ds = load_food101_dataset("train")

    dists = [1, 2, 4, 7, 10, 15]
    results = {}
    for dist in dists:
        ds = ds.map(lambda x: compression_fn(x, dist), batched=True, num_proc=8)
        results[dist] = sum(ds["size"]) / len(ds["size"])
        print(f"Done with distance {dist} - {results[dist]}")

    with open(FOOD101_SIZES, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    calc_jxl_sizes_from_files()
