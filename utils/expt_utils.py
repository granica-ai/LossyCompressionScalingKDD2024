"""Helper functions for experiments."""

import os
import socket
import logging
import sys
from typing import Tuple


def set_cuda_device():
    """Set the visible GPUs for training."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if socket.gethostname() == "aurora":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_logger(log_file: str) -> logging.Logger:
    """
    Create a logger that logs to a file and the console.

    Args:
        log_file: str, path to the log file

    Returns:
        logging.Logger: logger object
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    return logging.getLogger(__name__)


def binary_search_largest_below_function(
    func: callable, target: float, left: float, right: float, precision: float = 1e-6
) -> Tuple[float, float]:
    """
    Performs binary search to find the largest x for which func(x) is less than or equal to target.
    Assumes func is a monotonically increasing function.

    Args:
        func (callable): Monotonically increasing function
        target (float): Target value to search for
        left (float): Left bound of the search range
        right (float): Right bound of the search range
        precision (float): Precision of the result, defaults to 1e-6

    Returns:
        The largest x for which func(x) is less than or equal to target, within the given
            precision and the value of func at that point.
    """
    result = left  # lower bound
    result_value = None
    overshot = False

    while right - left > precision:
        mid = left + (right - left) / 2
        value = func(mid)
        logging.info(f"mid: {mid}, value: {value}")
        if value <= target:
            result = mid
            result_value = value
            left = mid + precision
        else:
            right = mid
            overshot = True

    if result_value is None:
        raise ValueError(
            f"Could not find a result within the given precision {precision}"
        )
    if not overshot:
        logging.warning(
            "Binary search did not find any values that exceeded the target. "
            "This may indicate a problem with the search range."
        )
    logging.info(
        f"Finished binary search with result {result} and value {result_value}"
    )
    return result, result_value
