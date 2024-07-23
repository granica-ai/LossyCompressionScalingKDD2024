"""Make the subsets for the scaling law fitting models for the iSAID dataset."""

import os
import json
import numpy as np

from utils.consts import ISAID_SUBSET_ROOT, ISAID_ANN_PATH


def isaid_scaling_training_subsets():
    """
    Create annotation files for subsets of the iSAID training set with numbers of images.

    Creates 4 annotation files with subsets of different sizes (20%, 40%, 60%, and 80% of the
    total number of images in the training set). Each successively larger subset contains the
    previous dataset plus 20% more images.
    """
    # Load the original COCO format JSON file
    with open(ISAID_ANN_PATH, "r") as file:
        coco_data = json.load(file)

    n = len(coco_data["images"])
    img_ids = sorted([img["id"] for img in coco_data["images"]])
    # Shuffling because not sure if the images are ordered in any way (does disrupt patches though)
    rng = np.random.default_rng(2024)
    rng.shuffle(img_ids)

    subset_fracs = np.arange(0.2, 0.9, 0.2)
    subset_sizes = np.array([n * x for x in subset_fracs], dtype=int)
    images_so_far = []
    annotations_so_far = []

    for frac, subset_size in zip(subset_fracs, subset_sizes):
        print(
            f"Creating subset with {subset_size} images... {len(images_so_far)} to {subset_size}."
        )
        # get new images and annotations
        new_ids = img_ids[len(images_so_far) : subset_size]
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
        save_path = os.path.join(ISAID_SUBSET_ROOT, f"train{frac:.1f}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as outfile:
            json.dump(subset_data, outfile, indent=4)
        print(f"Subset saved to {save_path}.")


if __name__ == "__main__":
    isaid_scaling_training_subsets()
