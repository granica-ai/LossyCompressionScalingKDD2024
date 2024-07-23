"""Empirical Ridge Regression Experiment"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
import pandas as pd
from itertools import product
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Union

sns.set_style("whitegrid")


@dataclass
class Expt:
    lambda_: float
    p: int
    q: int
    r: float
    tau: float


@dataclass
class ExptGrid:
    p: int
    q: int
    r: int
    tau: float
    ms: Union[List[int], np.ndarray]
    ns: Union[List[int], np.ndarray]
    extra_m: int
    niters: int
    alpha_options: Union[List[int], np.ndarray]

    def max_m(self):
        return max(self.ms) + self.extra_m


def generate_x(grid: ExptGrid) -> np.ndarray:
    """
    Create a random sample of x that meets the theory assumptions.

    Args:
        grid: ExptGrid object with the parameters for the experiment.

    Returns:
        np.ndarray: A random sample of x.
    """
    x = []
    for l in range(grid.max_m()):
        a = np.sqrt(3 * grid.p ** (-l))
        # variance == p^-l
        assert np.isclose(1 / 12 * (2 * a) ** 2, grid.p**-l)
        xi = np.random.uniform(low=-a, high=a, size=grid.q**l)
        x.append(xi)
    return np.hstack(x)


def calc_L(q: int, m: int) -> int:
    """
    Calculate the length of x given q and m.

    Args:
        q: The q parameter.
        m: The m parameter.

    Returns:
        int: The length of x.
    """
    return sum([q**l for l in range(m)])


def create_theta(grid: ExptGrid) -> np.ndarray:
    """
    Create the true theta vector that satisfies the necessary assumptions.

    Args:
        grid: ExptGrid object with the parameters for the experiment.

    Returns:
        np.ndarray: The theta vector.
    """
    theta = []
    for l in range(grid.max_m()):
        theta_l = np.ones(grid.q**l) * np.sqrt(grid.q**-l * grid.r**l)
        assert np.isclose(np.linalg.norm(theta_l) ** 2, grid.r**l)
        theta.append(theta_l)
    return np.hstack(theta)


def sample_xs(
    theta: np.ndarray, grid: ExptGrid, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample x and calculate y with the true theta plus noise.

    Args:
        theta: The true theta vector.
        grid: ExptGrid object with the parameters for the experiment.
        n: The number of samples to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x samples and y samples.
    """
    x_samples = np.zeros((int(n), int(calc_L(grid.q, grid.max_m()))))
    for i in range(n):
        x = generate_x(grid)
        x_samples[i, :] = x

    ys = x_samples @ theta + np.random.normal(scale=grid.tau, size=n)
    return x_samples, ys


def test_error(model: Ridge, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Calculate the test mean squared error of a model.

    Args:
        model: The model to evaluate.
        x_test: The x test samples.
        y_test: The y test samples.

    Returns:
        float: The test error.
    """
    yhat = model.predict(x_test)
    return np.mean((yhat - y_test) ** 2)


def fit_model(xtrain: np.ndarray, ytrain: np.ndarray, alpha: float = None) -> Ridge:
    """
    Fit a Ridge regression model to the data.

    Args:
        xtrain: The x training samples.
        ytrain: The y training samples.
        alpha: The regularization parameter.

    Returns:
        Ridge: The fitted model.
    """
    clf = Ridge(alpha=alpha, fit_intercept=False, tol=1e-6)
    clf.fit(xtrain, ytrain)
    return clf


def run_expt(
    grid: ExptGrid,
    plot: bool = False,
    seed: int = None,
    nm2alpha: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the empirical excess error for the model grid.

    Args:
        grid: ExptGrid object with the parameters for the experiment.
        plot: Whether to plot the regularization paths.
        seed: The random seed to use.
        nm2alpha: A dictionary of optimal alpha values for each n and m.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The empirical results and the optimal alpha values.
    """
    # create theta
    theta = create_theta(grid)
    kappa = np.log(grid.p) / np.log(grid.q)

    # generate x samples
    results = []
    alpha_df = []
    for i in tqdm(range(grid.niters)):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(i)

        x_train, y_train = sample_xs(theta=theta, grid=grid, n=max(grid.ns))
        x_test, y_test = sample_xs(theta=theta, grid=grid, n=10000)
        x_val, y_val = sample_xs(theta=theta, grid=grid, n=10000)

        # fit models
        test_errors = []
        for_plotting = {}
        for m, n in product(grid.ms, grid.ns):
            L = calc_L(grid.q, m)
            if nm2alpha is None:
                scaled_alpha_options = grid.alpha_options * n ** (
                    (kappa + 1) / (2 * kappa + 1)
                )
                alpha, errs = optimize_over_regularization(
                    scaled_alpha_options, x_train, y_train, x_val, y_val, n, L
                )
                record_regularizaion(alpha, alpha_df, i, for_plotting, m, n, errs)
            else:
                alpha = nm2alpha[(n, m)]
            model = fit_model(x_train[: int(n), :L], y_train[: int(n)], alpha)
            err = test_error(model, x_test[:, :L], y_test)
            test_errors.append(pd.Series({"m": m, "n": n, "L": L, "test_error": err}))
        results.append(pd.concat(test_errors, axis=1).T)
        if plot:
            plot_regularization(scaled_alpha_options, for_plotting, n)
    if nm2alpha is not None:
        return sum(results) / len(results), None
    return sum(results) / len(results), pd.concat(alpha_df, axis=1).T


def record_regularizaion(
    alpha: float,
    alpha_df: pd.DataFrame,
    i: int,
    for_plotting: dict,
    m: int,
    n: int,
    errs: np.ndarray,
):
    """
    Record the regularization path for a given n and m.

    Args:
        alpha: The optimal alpha value.
        alpha_df: The DataFrame to store the results.
        i: The iteration number.
        for_plotting: The dictionary to store the regularization paths.
        m: The m parameter.
        n: The n parameter.
        errs: The test errors for each alpha value.
    """
    for_plotting[(m, n)] = errs
    alpha_df.append(
        pd.Series(
            {
                "m": m,
                "n": n,
                "alpha": alpha,
                "iter": i,
                "alpha_idx": np.argmin(errs),
            }
        )
    )


def optimize_over_regularization(
    scaled_alpha_options: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n: int,
    L: int,
) -> Tuple[float, np.ndarray]:
    """
    Optimize over the regularization parameter for a given n and m.

    Args:
        scaled_alpha_options: The alpha values to test.
        x_train: The x training samples.
        y_train: The y training samples.
        x_val: The x validation samples.
        y_val: The y validation samples.
        n: The number of samples.
        L: The length of x.

    Returns:
        Tuple[float, np.ndarray]: The optimal alpha value and the test errors.
    """
    errs = []
    if scaled_alpha_options is None:
        scaled_alpha_options = 10 ** np.linspace(-1, 4, 15)

    for a in scaled_alpha_options:
        model = fit_model(x_train[: int(n), :L], y_train[: int(n)], a)
        err = test_error(model, x_val[:, :L], y_val)
        errs.append(err)

    amin = np.argmin(errs)
    alpha = scaled_alpha_options[amin]

    if amin == 0 or amin == len(scaled_alpha_options) - 1:
        print(f"Warning: alpha on boundary of search space - lower end={amin == 0}")
    return alpha, errs


def plot_regularization(alpha_options: np.ndarray, for_plotting: dict, n: int):
    """
    Plot the regularization paths for a given n.

    Args:
        alpha_options: The alpha values tested.
        for_plotting: The regularization paths.
        n: The number of samples.
    """
    for i, (k, v) in enumerate(for_plotting.items()):
        plt.plot(
            alpha_options,
            v,
            marker="o",
            label=f"n={k[1]}, m={k[0]}",
            color=sns.color_palette()[i],
        )
        plt.axvline(
            alpha_options[np.argmin(v)],
            color=sns.color_palette()[i],
            linestyle="--",
        )
    plt.xscale("log")
    plt.legend()
    plt.title(f"n={n}")
    plt.show()
