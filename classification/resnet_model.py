"""Model interface for ResNet50 image classification model."""

import os
import logging
from collections import namedtuple
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
import torchvision
import datasets
import torch.nn as nn
import PIL.Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from imagecodecs import jpegxl_encode, jpegxl_decode

from utils.jpegxl_transform import JPEGXLTransform
from utils.drr_estimation_tools import (
    get_images_for_target_size,
    get_target_size_from_subset,
)
from utils.consts import FOOD101_PATH


def load_food101_dataset(split: str) -> datasets.Dataset:
    """
    Load the Food101 dataset with specialized splits.

    Args:
        split: str, split to load, either train, val, or test

    Returns:
        datasets.Dataset, dataset
    """
    if split == "test":
        return datasets.load_dataset("food101", split="validation")
    ds = datasets.load_dataset("food101", split="train")
    splits = ds.train_test_split(test_size=0.2, seed=42)
    if split == "val":
        return splits["test"]
    elif split == "train":
        return splits["train"]
    else:
        raise ValueError("Invalid split")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        original_dataset: datasets.Dataset,
        transform: callable,
        extra_fn: callable = None,
        pre_transform: callable = None,
    ):
        """
        Initialize the CustomDataset.

        Args:
            original_dataset: datasets.Dataset, original dataset
            transform: callable, transformation function
            extra_fn: callable, function to apply to the image
            pre_transform: callable, pre-transformation function
        """
        self.original_dataset = original_dataset
        self.transform = transform
        self.extra_fn = extra_fn
        self.pre_transform = pre_transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        if self.pre_transform is not None:
            item["image"] = self.pre_transform(item["image"].convert("RGB"))
        if self.extra_fn is not None:
            item = self.extra_fn(item)
        item["image"] = self.transform(item["image"].convert("RGB"))
        return item


class ResNet50:
    def __init__(
        self,
        data,
        hyperparameters: dict,
        save_suffix: str = "",
        save_name_suffix: str = "",
    ) -> None:
        """
        Initialize the ResNet50 model.

        Args:
            data: DatasetData
            hyperparameters: dict, hyperparameters for the model
            save_suffix: str, suffix for the save path
            save_name_suffix: str, suffix for the save
        """
        self.data = data
        self.hyperparameters = hyperparameters
        self.save_suffix = save_suffix
        self.save_name_suffix = save_name_suffix
        self.save_path = os.path.join(FOOD101_PATH, save_suffix)
        os.makedirs(self.save_path, exist_ok=True)

    def get_unprocessed_data_loader(
        self,
        split: str,
        num_samples: int = None,
        distance: float = None,
        fraction: float = None,
        size: int = None,
        add_idx: bool = False,
        compress_before_crop: bool = False,
        specific_idxs: str = None,
        random_state: int = 42,
        required_subset_path: str = None,
    ) -> torch.utils.data.DataLoader:
        """
        Get a dataloader that loads a dataset that has not been processed.

        Args:
            num_samples: int, number of samples to use
            distance: float, Butteraugli distance
            fraction: float, fraction of the dataset to use
            size: int, target size of the images
            add_idx: bool, whether to add an index column
            compress_before_crop: bool, whether to compress before cropping
            specific_idxs: str, path to specific image indices to use
            random_state: int, random state for reproducibility
            required_subset_path: str, path to a subset of images that must be included

        Returns:
            torch.utils.data.DataLoader, dataloader
        """
        ds = self.get_dataset(
            split,
            num_samples,
            distance,
            fraction,
            size,
            add_idx,
            compress_before_crop,
            specific_idxs,
            random_state=random_state,
            required_subset_path=required_subset_path,
        )

        # set up transformations
        _transform = self._get_transform(
            distance,
            post_only=(size is not None and not compress_before_crop),
            compress_before_crop=compress_before_crop,
        )
        ds = CustomDataset(ds, _transform)

        # create dataloader
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.hyperparameters.get("batch_size", 32),
            shuffle=(split == "train"),
            num_workers=4,
        )
        return loader

    def get_dataset(
        self,
        split: str,
        num_samples: int = None,
        distance: float = None,
        fraction: float = None,
        size: int = None,
        add_idx: bool = False,
        compress_before_crop: bool = False,
        specific_idxs: str = None,
        random_state: int = 42,
        required_subset_path: str = None,
    ) -> datasets.Dataset:
        """
        Get the dataset.

        Args:
            num_samples: int, number of samples to use
            distance: float, Butteraugli distance
            fraction: float, fraction of the dataset to use
            size: int, target size of the images
            add_idx: bool, whether to add an index column
            compress_before_crop: bool, whether to compress before cropping
            specific_idxs: str, path to specific image indices to use
            random_state: int, random state for reproducibility
            required_subset_path: str, path to a subset of images that must be included

        Returns:
            datasets.Dataset, dataset
        """
        loaders = {
            "food101": load_food101_dataset,
        }
        ds = loaders[self.data.name](split)
        if (size is None and add_idx) or specific_idxs is not None:
            ds = ds.add_column("idx", range(len(ds)))

        if num_samples is not None:
            ds = self._get_num_samples_subset(num_samples, ds, random_state)
        elif fraction is not None:
            ds = self._get_fraction_subset(fraction, ds)
        elif size is not None:
            ds = self._get_size_subset(
                size,
                ds,
                distance,
                compress_before_crop,
                random_state=random_state,
                required_subset_path=required_subset_path,
            )
        elif specific_idxs is not None:
            ds = self._get_specific_idxs_subset(specific_idxs, ds)
        return ds

    def get_data_loader_file_compression(
        self,
        split: str,
        compression_file_path: str,
        num_samples: int = None,
        fraction: float = None,
        size: int = None,
        idx_check: bool = False,
        compress_before_crop: bool = False,
        specific_idxs: str = None,
        random_state: int = 42,
    ) -> torch.utils.data.DataLoader:
        """
        Get a dataloader that loads a dataset with file compression.

        Args:
            compression_file_path: str, path to a map of indices to Butteraugli distances
            num_samples: int, number of samples to use
            fraction: float, fraction of the dataset to use
            size: int, target size of the images
            idx_check: bool, whether to check the indices
            compress_before_crop: bool, whether to compress before cropping
            specific_idxs: str, path to specific image indices to use
            random_state: int, random state for reproducibility

        Returns:
            torch.utils.data.DataLoader, dataloader
        """
        logging.info(f"Loading data from {compression_file_path}")
        logging.info(f"{compress_before_crop=}")
        ds = self.get_dataset(
            split,
            num_samples=num_samples,
            distance=None,
            fraction=fraction,
            size=size,
            add_idx=True,
            compress_before_crop=compress_before_crop,
            specific_idxs=specific_idxs,
            random_state=random_state,
        )
        idxs = [ds[i]["idx"] for i in range(len(ds))]
        orig_idx2subset_idx = dict(zip(sorted(idxs), range(len(idxs))))

        if idx_check:
            expected_idxs = np.load(idx_check)
            assert set(expected_idxs) == set(
                idxs
            ), "Indices do not match expected subset"
            logging.info("Indices match expected subset")

        compression = np.load(compression_file_path)
        img_col = self.data.image_col

        def compression_fn(x):
            cl = compression[orig_idx2subset_idx[x["idx"]]]
            compr = jpegxl_decode(jpegxl_encode(x[img_col], distance=cl))
            x[img_col] = PIL.Image.fromarray(compr)
            return x

        # set up transformations
        if compress_before_crop:
            _transform = self._get_transform(distance=False)
            ds = CustomDataset(ds, _transform, extra_fn=compression_fn)
        else:
            pre_transform = self._get_transform(distance=False, pre_only=True)
            _transform = self._get_transform(distance=False, post_only=True)
            ds = CustomDataset(
                ds, _transform, extra_fn=compression_fn, pre_transform=pre_transform
            )

        # create dataloader
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.hyperparameters.get("batch_size", 32),
            shuffle=(split == "train"),
            num_workers=4,
        )
        return loader

    def _get_fraction_subset(
        self, fraction: float, ds: datasets.Dataset
    ) -> datasets.Dataset:
        """
        Get a specific fraction of the dataset.

        Args:
            fraction: float, fraction of the dataset to use
            ds: datasets.Dataset, dataset

        Returns:
            datasets.Dataset, subset of the dataset
        """
        if fraction > 1 - ds.features["label"].num_classes / len(ds):
            logging.warning(
                f"Number of samples too close to dataset size, using full dataset"
            )
        else:
            labels = ds["label"]
            idxs, _ = train_test_split(
                range(len(labels)),
                train_size=fraction,
                stratify=labels,
                random_state=42,
            )
            ds = torch.utils.data.Subset(ds, idxs)
        return ds

    def _get_num_samples_subset(
        self, num_samples: int, ds: datasets.Dataset, random_state: int = 42
    ) -> datasets.Dataset:
        """
        Get a specific number of samples from the dataset.

        Args:
            num_samples: int, number of samples to use
            ds: datasets.Dataset, dataset
            random_state: int, random state for reproducibility

        Returns:
            datasets.Dataset, subset of the dataset
        """
        if num_samples > len(ds) - ds.features["label"].num_classes:
            logging.warning(
                f"Number of samples ({num_samples}) too close to dataset size ({len(ds)}), using full dataset"
            )
        else:
            logging.info(f"Using random state {random_state} to generate_subset")
            labels = ds["label"]
            idxs, _ = train_test_split(
                range(len(labels)),
                train_size=num_samples,
                stratify=labels,
                random_state=random_state,
            )
            ds = torch.utils.data.Subset(ds, idxs)
        return ds

    def _get_size_subset(
        self,
        size: int,
        ds: datasets.Dataset,
        distance: float,
        compress_before_crop: bool,
        random_state: int = 42,
        required_subset_path: str = None,
    ) -> datasets.Dataset:
        """
        Get a subset of images that total a specific size in bytes.

        Args:
            size: int, target size of the images
            ds: datasets.Dataset, dataset
            distance: float, Butteraugli distance
            compress_before_crop: bool, whether to compress before cropping
            random_state: int, random state for reproducibility
            required_subset_path: str, path to a subset of images that must be included

        Returns:
            datasets.Dataset, subset of the dataset
        """
        x = namedtuple("FakeModelData", "dataset_data")(dataset_data=self.data)
        if not compress_before_crop:
            pre_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
            )
            ds = ds.map(lambda x: {"image": pre_transform(x["image"])}, num_proc=4)
            logging.info(
                "Resize and crop applied before looking for images of target size"
            )
        ds = ds.add_column("idx", range(len(ds)))
        if required_subset_path is not None:
            required_subset = np.load(required_subset_path)
            fixed_subset = self._get_specific_idxs_subset(required_subset, ds)
            extra_images = self._get_specific_idxs_subset(
                set(range(len(ds))) - set(required_subset), ds
            )
            subset = get_target_size_from_subset(
                fixed_subset.cast_column("image", datasets.Image(decode=False)),
                extra_imgs=extra_images.cast_column(
                    "image", datasets.Image(decode=False)
                ),
                target_size=size,
                ba=distance,
                model=x,
                image_col=self.data.image_col,
                from_img_path=False,
                random_state=random_state,
                epsilon=1e-4,
            )
        else:
            subset = get_images_for_target_size(
                ds.cast_column("image", datasets.Image(decode=False)),
                size,
                distance,
                x,
                self.data.image_col,
                from_img_path=False,
                from_preloaded=False,
                epsilon=1e-4,
                random_state=random_state,
            )
        return subset.cast_column("image", datasets.Image(decode=True))

    def _get_specific_idxs_subset(
        self, specific_idxs: np.ndarray, ds: datasets.Dataset
    ) -> datasets.Dataset:
        """
        Select a subset of the dataset based on specific indices.

        Args:
            specific_idxs: np.ndarray, specific indices to use
            ds: datasets.Dataset, dataset

        Returns:
            datasets.Dataset, subset of the dataset
        """
        ds = ds.filter(lambda x: x["idx"] in set(specific_idxs))
        logging.info(f"Specific indicies leaves {len(ds)} samples")
        return ds

    def _get_transform(
        self,
        distance: float,
        post_only: bool = False,
        pre_only: bool = False,
        compress_before_crop: bool = False,
    ) -> transforms.Compose:
        """
        Get the transformations for the dataset.

        Args:
            distance: float, Butteraugli distance
            post_only: bool, whether to only apply post-transformations
            pre_only: bool, whether to only apply pre-transformations
            compress_before_crop: bool, whether to compress before cropping

        Returns:
            transforms.Compose, transformations
        """
        # set up transformations
        transform_list = []
        if compress_before_crop:
            transform_list.append(JPEGXLTransform(distance=distance))

        if not post_only:
            transform_list += [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        if pre_only:
            logging.info(f"Pre transforms: {transform_list}")
            return transforms.Compose(transform_list)
        if not compress_before_crop and distance:
            transform_list.append(JPEGXLTransform(distance=distance))
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        logging.info(f"Transforms: {transform_list}")
        _transform = transforms.Compose(transform_list)
        return _transform

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        distance: float,
    ) -> torchvision.models.resnet.ResNet:
        """
        Train the model.

        Args:
            train_loader: torch.utils.data.DataLoader, dataloader
            distance: float, Butteraugli distance

        Returns:
            torchvision.models.resnet.ResNet, trained model
        """
        # create model
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        model.fc = nn.Linear(model.fc.in_features, self.data.num_classes)

        # create loss function
        criterion = nn.CrossEntropyLoss()

        # create optimizer
        lr = self.hyperparameters.get("lr", 1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        # train model
        num_epochs = self.hyperparameters.get("num_epochs", 3)
        for epoch in range(num_epochs):
            for sample in tqdm(train_loader):
                images = sample["image"].to(device)
                labels = sample[self.data.label_col].to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            logging.info(f"Epoch: {epoch}, Loss: {loss.item()}")

        # save model
        n = len(train_loader.dataset)
        save_path = self.save_path + self.get_model_name(distance, n)
        torch.save(model.state_dict(), save_path)
        logging.info(f"Saved model to {save_path}")
        return model

    def get_model_name(self, distance: float, num_examples: int) -> str:
        """
        Create the model name from the butteraugli distance and number of examples.

        Args:
            distance: float, Butteraugli distance
            num_examples: int, number of examples

        Returns:
            str, model file name
        """
        return f"resnet50_dist_{distance}_n_{num_examples}{self.save_name_suffix}.pth"

    def load_model(
        self, distance: float, num_examples: int, model_path: str = None
    ) -> torchvision.models.resnet.ResNet:
        """
        Load a model based on the Butteraugli distance and number of examples or model data.

        Args:
            distance: float, Butteraugli distance
            num_examples: int, number of examples
            model_path: str, path to the model

        Returns:
            torchvision.models.resnet.ResNet, model
        """
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.data.num_classes)
        if model_path is None:
            model_path = self.save_path + self.get_model_name(distance, num_examples)

        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError:
            state_dict = torch.load(model_path)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        return model

    def evaluate(
        self,
        model: torchvision.models.resnet.ResNet,
        loader: torch.utils.data.DataLoader,
        correct_arr: bool = False,
        calc_f1: bool = False,
    ) -> float:
        """
        Evaluate the model.

        Args:
            model: torchvision.models.resnet.ResNet, model
            loader: torch.utils.data.DataLoader, dataloader
            correct_arr: bool, whether to return a boolean array of prediction correctness
            calc_f1: bool, whether to calculate the F1 score

        Returns:
            float, accuracy
        """
        # evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        if correct_arr:
            correct = np.zeros(len(loader.dataset))
        correct_count = 0
        if calc_f1:
            all_predicted = []
            all_labels = []
        total = 0
        for sample in loader:
            images, labels = sample["image"].to(device), sample[self.data.label_col].to(
                device
            )
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()
            if correct_arr:
                correct[sample["idx"]] = (predicted == labels).cpu().numpy()
            if calc_f1:
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(
            f"Accuracy of the model on the {total} "
            "validation images: {} %".format(100 * correct_count / total)
        )

        if calc_f1:
            # Calculate F1 score
            f1 = f1_score(all_labels, all_predicted, average="weighted")
            print(
                f"F1 Score of the model on the {total} validation images: {f1 * 100} %"
            )
            return correct_count / total, f1

        if correct_arr:
            return correct_count / total, correct
        return correct_count / total
