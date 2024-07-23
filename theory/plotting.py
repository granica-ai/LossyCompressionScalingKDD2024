"""Plotting utils for the theory module."""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines


def add_lines_to_legend(
    ax: Axes,
    label_1: str,
    label_2: str,
    label_3: str = None,
    legend: bool = True,
    legend_kwargs: dict = {},
):
    """
    Add solid, dotted, and dashed lines to the legend.

    Args:
        ax: The axis object of the plot.
        label_1: The label for the first line.
        label_2: The label for the second line.
        label_3: The label for the third line.
        legend: Whether to add the custom lines to the legend.
        legend_kwargs: Keyword arguments for the legend
    """
    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Create custom legend artists
    line_A = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None", label=label_1, markersize=3
    )
    line_B = mlines.Line2D([], [], color="black", linestyle="-", label=label_2)

    # Add the custom legend artists to the handles and labels
    if legend:
        handles = handles[: len(handles) // 2] + [line_A, line_B]
        labels = labels[: len(labels) // 2] + [label_1, label_2]
    else:
        handles = [line_A, line_B]
        labels = [label_1, label_2]

    if label_3 is not None:
        line_C = mlines.Line2D([], [], color="black", linestyle=":", label=label_3)
        handles.append(line_C)
        labels.append(label_3)

    # Create a new legend with the updated handles and labels
    ax.legend(handles=handles, labels=labels, fontsize="x-small", **legend_kwargs)


def create_pivot_df(
    results: pd.DataFrame, col: str, comparison: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a pivot table from the results DataFrame.

    Args:
        results: The results DataFrame.
        col: The column to pivot.
        comparison: The comparison DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The results and comparison pivot tables.
    """
    result_arr = results.pivot(index="n", columns="L", values=col)
    result_arr.columns = result_arr.columns.astype(int)
    result_arr.index = result_arr.index.astype(int)

    if comparison is not None:
        comparison_arr = comparison.pivot(index="n", columns="L", values="excess_err")
        comparison_arr.columns = comparison_arr.columns.astype(int)
        comparison_arr.index = comparison_arr.index.astype(int)
    else:
        comparison_arr = None
    return result_arr, comparison_arr


def plot_theory_line(
    result_arr: pd.DataFrame, ax: Axes, color: str, fig: Figure, cbar_label: str
) -> Tuple[mcolors.Colormap, mcolors.LogNorm]:
    """
    Plot the theoretical lines on the plot.

    Args:
        result_arr: The results DataFrame.
        ax: The axis object of the plot.
        color: The color map to use.
        fig: The figure object of the plot.
        cbar_label: The label for the color bar.

    Returns:
        Tuple[mcolors.Colormap, mcolors.LogNorm]: The color map and normalization
    """
    if color is not None:
        cmap = plt.get_cmap(color)

        # Normalize with logarithmic normalization
        norm = mcolors.LogNorm(
            vmin=result_arr.columns.min(), vmax=result_arr.columns.max()
        )

        for i, col in enumerate(result_arr.columns):
            ax.plot(
                result_arr.index, result_arr[col], linewidth=1.5, color=cmap(norm(col))
            )

        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=ax, orientation="vertical", label=cbar_label, pad=0.01
        )
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        return cmap, norm

    else:
        result_arr.plot(ax=ax, linewidth=1.5)
        return None, None


def plot_empirical_line(
    comparison_arr: pd.DataFrame,
    ax: Axes,
    color: str,
    cmap: mcolors.Colormap,
    norm: mcolors.LogNorm,
):
    """
    Plot the empirical lines on the plot.

    Args:
        comparison_arr: The comparison DataFrame.
        ax: The axis object of the plot.
        color: The color map to use.
        cmap: The color map object.
        norm: The normalization object.
    """
    if color is not None:
        for i, col in enumerate(comparison_arr.columns):
            ax.plot(
                comparison_arr.index,
                comparison_arr[col],
                markersize=3,
                color=cmap(norm(col)),
                marker="o",
                lw=0,
            )
    else:
        comparison_arr.plot(
            ax=ax, markersize=3, color=sns.color_palette(), marker="o", lw=0
        )


def plot_curve_line(
    ax: Axes,
    curve: callable,
    popt: np.ndarray,
    color: str,
    cmap: mcolors.Colormap,
    norm: mcolors.LogNorm,
    result_arr: pd.DataFrame,
    x: np.ndarray,
    plot_l: bool = True,
):
    """
    Plot the fitted scaling curve on the plot.

    Args:
        ax: The axis object of the plot.
        curve: The curve function to plot.
        popt: The optimized parameters.
        color: The color map to use.
        cmap: The color map object.
        norm: The normalization object.
        result_arr: The results DataFrame.
        x: The x values to plot.
        plot_l: Whether to plot the curve as a function of L.
    """
    if color is not None:
        color_list = [cmap(norm(col)) for col in result_arr.columns]
    else:
        color_list = sns.color_palette()
    if plot_l:
        for i, l in enumerate(result_arr.columns):
            ax.plot(x, curve([x, l], *popt), linestyle=":", color=color_list[i])
    else:
        for i, n in enumerate(result_arr.columns):
            ax.plot(x, curve([n, x], *popt), linestyle=":", color=color_list[i])


def plot_results(
    results: pd.DataFrame,
    save_name: str = None,
    title: str = None,
    comparison: pd.DataFrame = None,
    curve: callable = None,
    popt: np.ndarray = None,
    col: str = "excess_err",
    legend: bool = True,
    color: str = None,
    legend_kwargs: dict = {},
    separate_plots: bool = False,
):
    """
    Plot the theoretical, empirical, and scaling theory results.

    Args:
        results: The results DataFrame.
        save_name: The name to save the plot.
        title: The title of the plot.
        comparison: The comparison DataFrame.
        curve: The curve function to plot.
        popt: The optimized parameters.
        col: The column to plot.
        legend: Whether to add a legend.
        color: The color map to use.
        legend_kwargs: Keyword arguments for the legend.
        separate_plots: Whether to plot the results on separate plots.
    """
    result_arr, comparison_arr = create_pivot_df(results, col, comparison)

    if separate_plots:
        fig = plt.figure(figsize=(4.5, 3.2))
        ax1 = plt.gca()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5))

    cmap, norm = plot_theory_line(result_arr, ax1, color, fig, cbar_label="Bits $L$")

    if comparison is not None:
        plot_empirical_line(comparison_arr, ax1, color, cmap, norm)
    if curve is not None:
        x = np.linspace(min(result_arr.index) - 10, max(result_arr.index) + 20, 100)
        plot_curve_line(ax1, curve, popt, color, cmap, norm, result_arr, x)

    ax1.set_xlabel("Number of Samples $n$")
    ax1.set_ylabel("Excess Test Error")
    ax1.loglog()

    if legend:
        ax1.legend(result_arr.columns, **legend_kwargs)
    else:
        ax1.legend([])
    if comparison is not None:
        add_lines_to_legend(
            ax1,
            "Empirical",
            "Exact",
            label_3="Scaling" if curve is not None else None,
            legend=legend,
            legend_kwargs=legend_kwargs,
        )

    if separate_plots:
        plt.tight_layout(pad=0.5)
        if save_name is not None:
            plt.savefig(save_name[0])
            plt.close()
        else:
            plt.show()
        fig = plt.figure(figsize=(4.5, 3.2))
        ax2 = plt.gca()

    if comparison is not None:
        result_arr = result_arr.loc[result_arr.index.isin(comparison_arr.index)]

    cmap, norm = plot_theory_line(
        result_arr.T, ax2, color, fig, cbar_label="Number of Samples $n$"
    )
    if comparison is not None:
        plot_empirical_line(comparison_arr.T, ax2, color, cmap, norm)
    if curve is not None:
        x = np.linspace(min(result_arr.columns) - 0.2, max(result_arr.columns) + 1, 100)
        plot_curve_line(
            ax2, curve, popt, color, cmap, norm, result_arr.T, x, plot_l=False
        )

    ax2.set_xlabel("Bits $L$")
    ax2.set_ylabel("Excess Test Error")
    ax2.loglog()

    if legend:
        ax2.legend(result_arr.index, **legend_kwargs)
    else:
        ax2.legend([])
    if comparison is not None:
        add_lines_to_legend(
            ax2,
            "Empirical",
            "Exact",
            label_3="Scaling" if curve is not None else None,
            legend=legend,
            legend_kwargs=legend_kwargs,
        )
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(pad=0.5)
    if save_name is not None:
        if separate_plots:
            plt.savefig(save_name[1])
        else:
            plt.savefig(save_name)
        plt.close()
    else:
        plt.show()
