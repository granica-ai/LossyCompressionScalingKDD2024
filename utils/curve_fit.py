"""Functions for fitting a curve to the data and plotting the results."""

from typing import Tuple
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def curve(x: tuple, a: float, b: float, alpha: float, beta: float) -> float:
    """a * n^-alpha + b * L^-beta"""

    return a * (x[0] ** -alpha) + b * (x[1] ** -beta)


def curve_plus_const(
    x: tuple, a: float, b: float, alpha: float, beta: float, c: float
) -> float:
    """a * n^-alpha + b * L^-beta + c"""
    return a * (x[0] ** -alpha) + b * (x[1] ** -beta) + c


def curve_multiplicative(
    x: tuple, a: float, alpha: float, beta: float, c: float
) -> float:
    """a * n^-alpha * L^-beta + c"""
    return a * (x[0] ** -alpha) * (x[1] ** -beta) + c


def calc_r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R^2 value of the fit."""
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(r"$R^2$" + f": {r2}")
    return r2


def melt_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Melt the DataFrame into the format needed for curve fitting."""
    data = (
        df.reset_index(names="num_points")
        .melt(var_name="px_per_img", value_name="test_error", id_vars="num_points")
        .dropna()
    )
    x = data[["num_points", "px_per_img"]].to_numpy()
    y = data.test_error.to_numpy()
    return x, y


def make_estimate_df(
    df: pd.DataFrame, params: np.ndarray, curve: callable
) -> pd.DataFrame:
    """Make a DataFrame of the estimates from the curve fit."""
    estimates = pd.DataFrame(columns=df.columns, index=df.index)
    for n in df.index:
        for L in df.columns:
            estimates.at[n, L] = curve([n, L], *params)
    return estimates


def plot_estimates(
    df: pd.DataFrame,
    estimates: pd.DataFrame,
    fit: str,
    model_dataset: str = "Food101 ResNet50",
):
    """Plot the estimates from the curve fit."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    df.sort_index().plot(style="o-", ax=axs[0], label=df.columns)
    cmap = dict(zip(df.columns, sns.color_palette()))
    estimates.plot(style=":", ax=axs[0], legend=False, color=cmap)
    axs[0].set_xlabel("Number of Samples")
    axs[0].set_ylabel("Test Error")
    axs[0].set_title(f"{model_dataset} Scaling\nCurve Fit: {fit}")
    axs[0].legend(
        title="Avg Bits / Img",
        loc="upper right",
        labels=df.columns,
        fontsize="x-small",
        title_fontsize="x-small",
    )

    # plot other side
    tmp = df.T
    tmp.index = tmp.index.astype(int)
    tmp.sort_index().plot(style="o-", ax=axs[1])
    cmap = dict(zip(df.index, sns.color_palette()))
    tmp2 = estimates.T
    tmp2.index = tmp2.index.astype(int)
    tmp2.sort_index().plot(style=":", ax=axs[1], label="_", color=cmap)
    axs[1].set_xlabel("Avg Bits per Image")
    axs[1].set_ylabel("Test Error")
    axs[1].set_title(f"{model_dataset} Scaling\nCurve Fit: {fit}")
    axs[1].legend(
        title="Num Samples",
        loc="upper right",
        labels=df.index,
        fontsize="x-small",
        title_fontsize="x-small",
    )
    plt.tight_layout()
    plt.show()


def try_available_fits(df: pd.DataFrame, model_dataset: str = "Food101 ResNet50"):
    """Try fitting the data with different curve functions and plot results."""
    x, y = melt_data(df)
    param_bounds = (
        (0, 0, 0, 0),
        (np.inf, np.inf, np.inf, np.inf),
    )
    param_bounds_plus_const = (
        (0, 0, 0, 0, 0),
        (np.inf, np.inf, np.inf, np.inf, np.inf),
    )
    param_bounds_multiplicative = (
        (0, 0, 0, 0),
        (np.inf, np.inf, np.inf, np.inf),
    )

    fit = r"$a \cdot n^{-\alpha} + b \cdot L^{-\beta}$"
    popt_orig = try_fit(df, x, y, fit, curve, {"bounds": param_bounds}, model_dataset)

    fit = r"$a \cdot n^{-\alpha} + b \cdot L^{-\beta} + c$"
    try_fit(
        df,
        x,
        y,
        fit,
        curve_plus_const,
        {"bounds": param_bounds_plus_const},
        model_dataset=model_dataset,
    )

    fit = r"$a \cdot n^{-\alpha} + b \cdot L^{-\beta} + c$ with $p_0$"
    try_fit(
        df,
        x,
        y,
        fit,
        curve_plus_const,
        {"bounds": param_bounds_plus_const, "p0": np.append(popt_orig, 0)},
        model_dataset=model_dataset,
    )

    np.random.seed(0)
    fit = r"$a \cdot n^{-\alpha} \cdot L^{-\beta} + c$"
    try_fit(
        df,
        x,
        y,
        fit,
        curve_multiplicative,
        {"bounds": param_bounds_multiplicative, "p0": np.random.rand(4) * 10},
        model_dataset=model_dataset,
    )


def try_fit(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    fit: str,
    curve_fn: callable,
    fit_kwargs: dict = {},
    model_dataset: str = "Food101 ResNet50",
) -> np.ndarray:
    """Try fitting the data with a curve function and plot results."""
    try:
        popt, pcov, _, mesg, _ = curve_fit(
            curve_fn, x.T, y, **fit_kwargs, full_output=True
        )
        perr = np.sqrt(np.diag(pcov))
        print("Curve fit:", mesg)
        print("Format: " + fit)
        print("[" + ", ".join([str(j) for j in popt]) + "]")
        print(f"Uncertainty: {perr}")
        estimates = make_estimate_df(df, popt, curve_fn)
        calc_r_squared(y, curve_fn(x.T, *popt))
        plot_estimates(df, estimates, fit, model_dataset=model_dataset)

    except RuntimeError:
        print(fit + " failed")
    return popt
