"""Make n subsets for scaling curve fitting models for the Cityscapes dataset."""

import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split

from utils.consts import CITYSCAPES_DATA_ROOT, CITYSCAPES_SUBSET_ROOT


def cityscapes_scaling_training_subsets():
    """
    Create files for subsets of the cityscapes training set with numbers of images.

    Creates 4 files with subsets of different sizes (20%, 40%, 60%, and 80% of the
    total number of images in the training set).
    """
    os.makedirs(CITYSCAPES_SUBSET_ROOT, exist_ok=True)

    subset = glob.glob(os.path.join(CITYSCAPES_DATA_ROOT, "train", "*", "*.png"))
    train_root = os.path.join(CITYSCAPES_DATA_ROOT, "train")
    train_suffix = "_leftImg8bit.png"
    subset = [x[len(train_root) + 1 : -len(train_suffix)] for x in subset]
    num_train = len(subset)
    print(f"Total number of images: {num_train}")
    subset_fractions = np.arange(0.2, 0.9, 0.2)[::-1]
    subset_counts = [int(num_train * x) for x in subset_fractions]

    full_ds_path = os.path.join(CITYSCAPES_SUBSET_ROOT, "train1.0.txt")
    with open(full_ds_path, "w") as f:
        f.write("\n".join(subset))

    for frac, count in zip(subset_fractions, subset_counts):
        subset = train_test_split(subset, train_size=count, random_state=42)[0]
        print(f"Subset {frac:.1f} has {len(subset)} images")
        subset_path = os.path.join(CITYSCAPES_SUBSET_ROOT, f"train{frac:.1f}.txt")
        with open(subset_path, "w") as f:
            f.write("\n".join(subset))


if __name__ == "__main__":
    cityscapes_scaling_training_subsets()
