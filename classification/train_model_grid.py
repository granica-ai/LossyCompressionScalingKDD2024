"""Models to fit scaling curves for the food101 dataset."""

from utils.expt_utils import set_cuda_device

set_cuda_device()

import logging
import os
import numpy as np
from datetime import datetime
import dataclasses
import pandas as pd
from classification.resnet_model import ResNet50
from utils.consts import DEFAULT_FOOD101_PARAMS, COMPRESSION_LEVELS, FOOD101_PATH
from utils.expt_utils import make_logger


@dataclasses.dataclass
class DatasetData:
    """Dataclass for dataset information."""

    name: str
    num_classes: int
    size: int
    stratify: bool
    image_col: str = "image"
    label_col: str = "label"


def food101_zero_drr_estimator():
    """Estimate the Butteraugli distance that achievees zero DRR for the Food101 dataset."""
    return (1 - 0.71150908) / 0.58897115


@dataclasses.dataclass
class Food101(DatasetData):
    """Dataclass for Food101 dataset."""

    name: str = "food101"
    num_classes: int = 101
    size: int = 60600
    stratify: bool = True
    zero_drr_estimator: callable = food101_zero_drr_estimator


class ScalingGrid:
    def __init__(
        self,
        data: DatasetData,
        hyperparameters: dict,
        compression_levels: list = None,
        num_points: list = None,
        force_continue_path: str = None,
        save_suffix: str = "",
        compress_before_crop: bool = False,
    ):
        """
        Initialize the scaling grid.

        Args:
            data: DatasetData
            hyperparameters: dict, hyperparameters for the model
            compression_levels: list, Butteraugli distances
            num_points: list, number of points to sample
            force_continue_path: str, path to continue from
            save_suffix: str, suffix for the save path
            compress_before_crop: bool, whether to compress before cropping
        """
        self.data = data
        self.force_continue_path = force_continue_path
        self.compress_before_crop = compress_before_crop
        self.suffix = save_suffix
        if num_points is None:
            self.num_points = [int(data.size * x) for x in np.arange(0.2, 1.1, 0.2)]
        else:
            self.num_points = num_points
        self.compression_levels = compression_levels or COMPRESSION_LEVELS

        self.set_up_saving(save_suffix)

        self.hyperparameters = hyperparameters
        self.logger.info(f"Hyperparameters: {hyperparameters}")
        self.logger.info(f"Number of points: {self.num_points}")
        self.logger.info(f"Compression levels: {self.compression_levels}")

        self.model = ResNet50(
            data=data, hyperparameters=hyperparameters, save_suffix=save_suffix
        )

    def set_up_logging(self, timestamp: str) -> None:
        """Set up logging."""
        log_file = os.path.join(self.work_dir, f"{timestamp}-log.log")
        self.logger = make_logger(log_file)

    def set_up_saving(self, save_suffix: str) -> None:
        """
        Set up saving, resuming from checkpoint if specified.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.work_dir = os.path.join(FOOD101_PATH, save_suffix)
        self.set_up_logging(timestamp)

        if self.force_continue_path is not None:
            acc_path = os.path.join(self.work_dir, self.force_continue_path)
            self.accuracies = pd.read_csv(acc_path, index_col=0)
            self.logger.info(f"Continuing from {acc_path}")
        else:
            self.accuracies = pd.DataFrame(
                columns=[str(x) for x in self.compression_levels],
                index=self.num_points,
            )

    def run(self) -> None:
        """Train models in grid and evaluate."""
        self.logger.info(f"Running on {self.data.name}")

        test_loader = self.model.get_unprocessed_data_loader("test")

        for n in self.num_points:
            for dist in self.compression_levels:
                self.logger.info(
                    f"\n====== Number of points: {n}, Distance: {dist} ======\n"
                )
                filepath = os.path.join(
                    self.work_dir, self.model.get_model_name(dist, n)
                )
                if (
                    str(dist) in self.accuracies.columns
                    and n in self.accuracies.index
                    and not pd.isna(self.accuracies.at[n, str(dist)])
                ):
                    self.logger.info("Already computed, continuing...")
                    continue
                elif os.path.isfile(filepath):
                    self.logger.info("Model exists, loading from file...")
                    model = self.model.load_model(dist, n)
                else:
                    self.logger.info("Model does not exist, training...")
                    train_loader = self.model.get_unprocessed_data_loader(
                        "train",
                        num_samples=n,
                        distance=dist,
                        compress_before_crop=self.compress_before_crop,
                    )
                    model = self.model.train(train_loader, distance=dist)

                self.logger.info("Evaluating...")
                acc = self.model.evaluate(model, test_loader)
                self.logger.info(f"Accuracy: {acc}")
                self.accuracies.at[n, str(dist)] = acc

                save_path = os.path.join(self.work_dir, "accuracies.csv")
                logging.info(f"Saving results to {save_path}")
                self.accuracies.to_csv(save_path)  # save partial results

        print(self.accuracies.to_string())

    def evaluate(self, calc_f1: bool = True) -> None:
        """
        Evaluate the models, potentially on an alternative metric.

        Args:
            calc_f1: bool, whether to calculate the F1 score
        """
        self.logger.info(f"Running on {self.data.name}")

        test_loader = self.model.get_unprocessed_data_loader("test")

        for n in self.num_points:
            for dist in self.compression_levels:
                self.logger.info(
                    f"\n====== Number of points: {n}, Distance: {dist} ======\n"
                )
                filepath = os.path.join(
                    FOOD101_PATH, f"resnet50_dist_{dist}_n_{n}{self.suffix}.pth"
                )

                if not os.path.isfile(filepath):
                    self.logger.info("Model does not exist, skipping...")
                    continue

                self.logger.info("Model exists, loading from file...")
                model = self.model.load_model(dist, n, model_path=filepath)

                self.logger.info("Evaluating...")
                metrics = self.model.evaluate(model, test_loader, calc_f1=calc_f1)
                if calc_f1:
                    acc, f1 = metrics
                    self.logger.info(f"Accuracy: {acc}")
                    self.logger.info(f"F1 score: {f1}")
                    self.accuracies.at[n, str(dist)] = acc
                    self.accuracies.at[n, f"{dist}_f1"] = f1
                else:
                    self.logger.info(f"Accuracy: {metrics}")
                    self.accuracies.at[n, str(dist)] = metrics

                save_path = os.path.join(self.work_dir, "accuracies.csv")
                if calc_f1:
                    save_path = os.path.join(self.work_dir, "accuracies_f1.csv")
                logging.info(f"Saving results to {save_path}")
                self.accuracies.to_csv(save_path)  # save partial results

        print(self.accuracies.to_string())

    def evaluate_test_compression(self, n: int = 60600) -> None:
        """
        Evaluate the models on the test set with different compression levels.

        Args:
            n: int, number of points to evaluate with
        """
        results = []
        self.logger.info(f"Running on {self.data.name}")

        for train_dist in self.compression_levels:
            for test_dist in self.compression_levels:
                self.logger.info(
                    f"\n====== Train Distance: {train_dist} Test Distance: {test_dist} ======\n"
                )
                filepath = os.path.join(
                    FOOD101_PATH, f"resnet50_dist_{train_dist}_n_{n}{self.suffix}.pth"
                )

                if not os.path.isfile(filepath):
                    self.logger.info("Model does not exist, skipping...")
                    continue

                self.logger.info("Model exists, loading from file...")
                model = self.model.load_model(train_dist, n, model_path=filepath)

                self.logger.info("Evaluating...")
                test_loader = self.model.get_unprocessed_data_loader(
                    "test",
                    distance=test_dist,
                    compress_before_crop=self.compress_before_crop,
                )
                acc = self.model.evaluate(model, test_loader, calc_f1=False)
                self.logger.info(f"Accuracy: {acc}")
                results.append((train_dist, test_dist, acc))

                save_path = os.path.join(self.work_dir, "test_set_compression.csv")
                logging.info(f"Saving results to {save_path}")
                result_df = pd.DataFrame(
                    results, columns=["train_dist", "test_dist", "accuracy"]
                )
                result_df.to_csv(save_path)  # save partial results

        print(result_df.to_string())


if __name__ == "__main__":
    dataset = Food101()

    for i in range(1, 5):
        num_points = [int(dataset.size * x) for x in [0.05]] + [
            int(dataset.size * x) for x in np.arange(0.2, 1.1, 0.2)
        ]
        suffix = f"_iter{i}"
        grid = ScalingGrid(
            dataset,
            hyperparameters=DEFAULT_FOOD101_PARAMS,
            save_suffix=suffix,
            num_points=num_points,
            compression_levels=COMPRESSION_LEVELS,
            compress_before_crop=True,
        )
        grid.run()
        grid.evaluate(calc_f1=True)
        grid.evaluate_test_compression()
