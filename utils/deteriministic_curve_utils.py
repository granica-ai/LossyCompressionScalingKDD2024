import logging
from typing import List, Tuple
import numpy as np
import simplejson as json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


def estimate_opt_n_l(
    s_range: List[int], scaling_curve: callable
) -> Tuple[List[float], List[float]]:
    """
    Estimate optimal n and L for a range of s values given a scaling curve.

    Args:
        s_range: List[int], storage sizes in bytes
        scaling_curve: callable, scaling curve function

    Returns:
        Tuple[List[float], List[float]], optimal n and L values
    """
    logging.info(f"Using s_range: {s_range}")
    opt_ls = []
    opt_ns = []
    for s in s_range:
        fun = scaling_curve(s)
        x0 = 1000
        result = minimize(fun, x0, method="Nelder-Mead")
        logging.info(f"{s=:,}, n={result.x[0]:,.1f}, L={s / result.x[0]:,.1f}")
        opt_ls.append(s / result.x[0])
        opt_ns.append(result.x[0])
    return opt_ns, opt_ls


def get_dist2l(sizes_path) -> dict:
    """
    Get the mapping from Butteraugli distance to L.

    Returns:
        dict, mapping from Butteraugli distance to L
    """
    with open(sizes_path, "r") as f:
        dist2l = json.load(f)
    dist2l = {int(k): v for k, v in dist2l.items()}
    return dist2l


def plot_compression_levels(
    opt_ls: List[float],
    dist2l: dict,
    opt_ns: List[float],
    curve: callable,
    popt: np.ndarray,
):
    """
    Plot the Butteraugli distance vs L data and the fitted curve.

    Args:
        opt_ls: List[float], optimal L values
        dist2l: dict, mapping from Butteraugli distance to L
        opt_ns: List[float], optimal n values
        curve: callable, curve function
        popt: np.ndarray, curve parameters
    """
    x = np.linspace(1, 15, 1000)
    plt.plot(dist2l.keys(), dist2l.values(), label="Empirical", lw=2)
    plt.plot(x, curve(np.array(x), *popt), label="Curve Fit", lw=2)
    for i, (l, n) in enumerate(zip(opt_ls, opt_ns)):
        plt.axhline(
            l,
            color=sns.color_palette()[3],
            label=("Opt $L$ for $s$ Values" if i == 0 else "_"),
            lw=1,
        )
        if n == min(opt_ns) or n == max(opt_ns):
            plt.text(0.5, l, f"n={n:,.0f}", ha="left", va="center")
    plt.title("L vs Compression Level")
    plt.xlabel("Butteraugli Distance")
    plt.ylabel("L")
    plt.legend()
    plt.show()
