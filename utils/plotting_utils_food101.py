import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import simplejson as json
import re
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata

from utils.consts import FOOD101_PATH, FOOD101_SIZES, FOOD101_SS


def get_food101_df():
    err = pd.DataFrame()
    for i in range(5):
        df = pd.read_csv(
            os.path.join(FOOD101_PATH, f"cmprs_first_iter{i}/accuracies.csv"),
            index_col=0,
        )
        df = 1 - df
        df.reset_index(inplace=True, names=["L"])
        df["iter"] = i
        err = pd.concat([err, df])

    with open(FOOD101_SIZES, "r") as f:
        food101dist2l = json.load(f)
    food101dist2l = {int(k): v for k, v in food101dist2l.items()}
    err.columns = [
        int(round(food101dist2l[int(x)])) if x not in ["L", "iter"] else x
        for x in df.columns
    ]
    df = err.groupby("L").mean(numeric_only=True).drop(columns=["iter"])
    df.index = df.index.astype(int)
    return df


def food101_scaling_n(df, curve, ns):
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")

    # Normalize cmapusing logarithmic normalization
    norm = mcolors.LogNorm(vmin=df.columns.min(), vmax=df.columns.max())

    for idx, row in df.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in df.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$")
    plt.ylabel(r"Test Error $Err_\text{test}$")
    plt.legend(
        title="Avg Bytes / Img",
        loc="upper right",
        labels=[f"{x:,}" for x in df.columns],
        fontsize=8,
        title_fontsize=10,
    )
    plt.loglog()
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6])
    plt.xticks([3e3, 1e4, 3e4, 6e4], ["3k", "10k", "30k", "60k"])
    plt.tight_layout()
    plt.show()


def food101_scaling_l(df, curve, ls):
    plt.figure(figsize=(4.5, 3.5))
    tmp = df.T
    tmp.index = tmp.index.astype(int)

    cmap = plt.get_cmap("crest")

    # Normalize cmap using logarithmic normalization
    norm = mcolors.Normalize(vmin=tmp.columns.min(), vmax=tmp.columns.max())

    for idx, row in tmp.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for idx in df.index:
        plt.plot(ls, curve(idx, ls), color=cmap(norm(idx)), label=idx, linestyle=":")
    plt.xlabel(r"Average Bytes per Image $L$")
    plt.ylabel(r"Test Error $Err_\text{test}$")

    plt.legend(
        title="Num Samples",
        loc=(0.5, 0.55),
        labels=[f"{x:,}" for x in df.index],
        fontsize=8,
        title_fontsize=10,
    )
    plt.loglog()
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6])
    plt.xticks([6e3, 1e4, 2e4, 3e4, 4e4], ["6k", "10k", "20k", "30k", "40k"])
    plt.tight_layout()
    plt.show()


def food101_scaling_plots():
    a, b, alpha, beta = [
        6.706939196262124,
        1355.2887767756654,
        0.33330403247197804,
        1.0613126086527451,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta)

    ns = np.linspace(2900, 61000, 100)
    ls = np.linspace(5000, 40000, 100)

    df = get_food101_df()
    food101_scaling_n(df, curve, ns)
    food101_scaling_l(df, curve, ls)


def extract_accuracies(file_path):
    accuracies = []
    accuracy_pattern = re.compile(r"Accuracy: (\d+\.\d+)")

    with open(file_path, "r") as file:
        for line in file:
            match = accuracy_pattern.search(line)
            if match:
                # Convert the accuracy to a float and append it to the list
                accuracies.append(float(match.group(1)))

    return accuracies


def food101_opt_s_curve():
    # Opt s
    det = extract_accuracies(
        os.path.join(FOOD101_PATH, "deterministic", "2024-01-20.log")
    )
    det_err = np.array([1 - x for x in det])
    det2 = extract_accuracies(
        os.path.join(FOOD101_PATH, "deterministic", "2024-01-28.log")
    )
    det2_err = np.array([1 - x for x in det2])
    det_err = (det_err + det2_err[:5] + det2_err[5:]) / 3

    # Uncompressed
    unc = extract_accuracies(
        os.path.join(FOOD101_PATH, "uncompressed", "2024-01-22.log")
    )
    unc_err1 = np.array([1 - x for x in unc])
    unc = extract_accuracies(
        os.path.join(FOOD101_PATH, "uncompressed", "2024-01-26.log")
    )
    unc_err2 = np.array([1 - x for x in unc])
    unc_err = (unc_err1 + unc_err2[:5] + unc_err2[5:]) / 3

    # Randomized
    rand_max15_1 = extract_accuracies(
        os.path.join(FOOD101_PATH, "randomized_max15", "2024-01-24_max15.log")
    )
    rand_max15_2 = extract_accuracies(
        os.path.join(FOOD101_PATH, "randomized_max15", "2024-01-29_max15_seed0.log")
    )
    rand_max15_3 = extract_accuracies(
        os.path.join(FOOD101_PATH, "randomized_max15", "2024-01-29_max15_seed1.log")
    )
    rand_max15 = (
        np.array(rand_max15_1) + np.array(rand_max15_2) + np.array(rand_max15_3)
    ) / 3
    rand_max15_err = [1 - x for x in rand_max15]

    plt.figure(figsize=(4.5, 3.5))
    plt.plot(FOOD101_SS, det_err, "o-", label="Scaling Curve Opt")
    plt.plot(FOOD101_SS, unc_err, "o-", label="Original Format")
    plt.plot(FOOD101_SS, rand_max15_err, "o-", label="Randomized")  # Max 15"
    plt.xlabel(r"Storage Size $s$ (Bytes)")
    plt.ylabel(r"Test Error $Err_\text{test}$ (% Incorrect)")
    plt.legend()
    plt.loglog()
    plt.tight_layout()
    plt.show()


def extract_s_n_size(log_file_path, uncompressed=False):
    if uncompressed:
        s_pattern = re.compile(r"Trying to find images for target size ([\d,]+)")
        n_pattern = re.compile(r"Number of points in uncompressed subset: (\d+)")
    else:
        s_pattern = re.compile(r"Target s: ([\d,]+)")
        n_pattern = re.compile(r"idxs_s[\d]+_n(\d+).npy")

    s_values, n_values, size_values = [], [], []

    with open(log_file_path, "r") as file:
        for line in file:
            s_match = s_pattern.search(line)
            if s_match:
                s_values.append(int(s_match.group(1).replace(",", "")))

            n_match = n_pattern.search(line)
            if n_match:
                n_values.append(int(n_match.group(1)))

    return {"s_values": s_values, "n_values": n_values}


def food101_opt_n(compr, uncompr):
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        compr["s_values"],
        compr["n_values"],
        label="Scaling Curve Opt",
        marker="o",
    )
    plt.plot(
        uncompr["s_values"],
        uncompr["n_values"],
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Number of Samples $n$", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks([5e7, 1e8, 2e8, 5e8], ["50", "100", "200", "500"], fontsize=12)
    plt.yticks(
        [1e3, 3e3, 1e4, 3e4, 6e4], ["1k", "3k", "10k", "30k", "60k"], fontsize=12
    )
    plt.tight_layout()
    plt.show()


def food101_opt_l(compr, uncompr):
    plt.figure(figsize=(4.5, 3.5))
    avg_l = np.divide(compr["s_values"], compr["n_values"])
    plt.plot(
        compr["s_values"],
        avg_l,
        label="Scaling Curve Opt",
        marker="o",
    )
    unc_avg_l = np.divide(uncompr["s_values"], uncompr["n_values"])
    plt.plot(
        uncompr["s_values"],
        unc_avg_l,
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Avg Bytes per Image $L$ (KB)", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks([5e7, 1e8, 2e8, 5e8], ["50", "100", "200", "500"], fontsize=12)
    plt.yticks([6e3, 1e4, 2e4, 3e4, 4e4], ["6", "10", "20", "30", "40"], fontsize=12)
    plt.tight_layout()
    plt.show()


def food101_opt_n_l_plots():
    log_file_path = os.path.join(
        FOOD101_PATH, "deterministic", "opt_compression_dists.log"
    )
    compr = extract_s_n_size(log_file_path)

    log_file_path = os.path.join(FOOD101_PATH, "uncompressed", "2024-01-22.log")
    uncompr = extract_s_n_size(log_file_path, uncompressed=True)
    food101_opt_n(compr, uncompr)
    food101_opt_l(compr, uncompr)


def food101_naive_compression_plot():
    # Opt s
    det = extract_accuracies(
        os.path.join(FOOD101_PATH, "deterministic", "2024-01-20.log")
    )
    det_err = np.array([1 - x for x in det])
    det2 = extract_accuracies(
        os.path.join(FOOD101_PATH, "deterministic", "2024-01-28.log")
    )
    det2_err = np.array([1 - x for x in det2])
    det_err = (det_err + det2_err[:5] + det2_err[5:]) / 3

    # Uncompressed
    unc = extract_accuracies(
        os.path.join(FOOD101_PATH, "uncompressed", "2024-01-22.log")
    )
    unc_err1 = np.array([1 - x for x in unc])
    unc = extract_accuracies(
        os.path.join(FOOD101_PATH, "uncompressed", "2024-01-26.log")
    )
    unc_err2 = np.array([1 - x for x in unc])
    unc_err = (unc_err1 + unc_err2[:5] + unc_err2[5:]) / 3

    ba3_subset_files = [
        os.path.join(
            FOOD101_PATH, "non_opt_compressed", "2024-02-02_ba3_subset_seed0.log"
        ),
        os.path.join(
            FOOD101_PATH, "non_opt_compressed", "2024-02-02_ba3_subset_seed1.log"
        ),
        os.path.join(FOOD101_PATH, "non_opt_compressed", "2024-02-03_ba3_subset.log"),
    ]

    ba3_subset = np.vstack([extract_accuracies(f) for f in ba3_subset_files]).mean(
        axis=0
    )
    ba3_subset_err = [1 - x for x in ba3_subset]

    ba8_subset_files = [
        os.path.join(
            FOOD101_PATH, "non_opt_compressed", "2024-01-30_ba8_subset_seed0.log"
        ),
        os.path.join(
            FOOD101_PATH, "non_opt_compressed", "2024-01-31_ba8_subset_seed1.log"
        ),
        os.path.join(FOOD101_PATH, "non_opt_compressed", "2024-02-01_ba8_subset.log"),
    ]

    ba8_subset = np.vstack([extract_accuracies(f) for f in ba8_subset_files]).mean(
        axis=0
    )
    ba8_subset_err = [1 - x for x in ba8_subset]

    ba13_subset_files = [
        os.path.join(
            FOOD101_PATH, "non_opt_compressed/2024-02-01_ba13_subset_seed0.log"
        ),
        os.path.join(
            FOOD101_PATH, "non_opt_compressed/2024-02-01_ba13_subset_seed1.log"
        ),
        os.path.join(FOOD101_PATH, "non_opt_compressed/2024-02-02_ba13_subset.log"),
    ]
    ba13_subset = np.vstack([extract_accuracies(f) for f in ba13_subset_files]).mean(
        axis=0
    )
    ba13_subset_err = [1 - x for x in ba13_subset]

    plt.figure(figsize=(4.5, 3.5))
    plt.plot(FOOD101_SS, det_err, "o-", label="Scaling Curve Opt", zorder=10)
    plt.plot(FOOD101_SS[: len(unc_err)], unc_err, "o-", label="Original Format")
    full_ds_size = (
        {  # found at /mnt/enigma0/km/results/scaling/food101/subset_sizes.json
            "3": 1265834716 / 60600,
            "8": 548005346 / 60600,
            "13": 382414373 / 60600,
        }
    )
    plt.plot(
        FOOD101_SS, ba3_subset_err, "o:", label=r"$L =$" + f" {full_ds_size['3']:,.0f}"
    )
    plt.plot(
        FOOD101_SS, ba8_subset_err, "o:", label=r"$L =$" + f" {full_ds_size['8']:,.0f}"
    )
    plt.plot(
        FOOD101_SS,
        ba13_subset_err,
        "o:",
        label=r"$L =$" + f" {full_ds_size['13']:,.0f}",
    )

    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=11)
    plt.ylabel(r"$Err_\text{test}$ (% Incorrect)", fontsize=11)
    plt.legend()
    plt.loglog()
    plt.xticks([5e7, 1e8, 2e8, 5e8], ["50", "100", "200", "500"])
    plt.yticks([0.20, 0.3, 0.4, 0.5, 0.6], ["20%", "30%", "40%", "50%", "60%"])
    plt.tight_layout()
    plt.show()


def get_food101_alt_metric_df():
    err = pd.DataFrame()
    for i in range(5):
        df = pd.read_csv(
            os.path.join(FOOD101_PATH, f"cmprs_first_iter{i}/accuracies_f1.csv"),
            index_col=0,
        )
        df = df[[x for x in df.columns if "f1" in x]]
        df.columns = [int(x[:-3]) for x in df.columns]
        df.drop(index=[6060], inplace=True)
        df = 1 - df
        df.reset_index(inplace=True, names=["L"])
        df["iter"] = i
        err = pd.concat([err, df])

    with open(FOOD101_SIZES, "r") as f:
        food101dist2l = json.load(f)
    food101dist2l = {int(k): v for k, v in food101dist2l.items()}
    err.columns = [
        int(round(food101dist2l[int(x)])) if x not in ["L", "iter"] else x
        for x in df.columns
    ]
    df = err.groupby("L").mean(numeric_only=True).drop(columns=["iter"])
    df.index = df.index.astype(int)
    return df


def food101_alt_metric_scaling_n(df, curve, ns):
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")

    # Normalize your data using logarithmic normalization
    norm = mcolors.LogNorm(vmin=df.columns.min(), vmax=df.columns.max())

    for idx, row in df.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in df.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$", fontsize=12)
    plt.ylabel(r"$Err_\text{test}$ (F1)", fontsize=12)
    plt.legend(
        title="Avg Bytes / Img",
        loc="lower left",
        labels=[f"{x:,}" for x in df.columns],
        fontsize=10,
        title_fontsize=10,
    )
    plt.loglog()
    plt.xticks([3e3, 1e4, 3e4, 6e4], ["3k", "10k", "30k", "60k"], fontsize=12)
    plt.yticks([0.2, 0.3, 0.4, 0.6], ["0.2", "0.3", "0.4", "0.6"], fontsize=12)
    plt.tight_layout()
    plt.show()


def food101_alt_metric_scaling_l(df, curve, ls):
    plt.figure(figsize=(4.5, 3.5))
    tmp = df.T
    tmp.index = tmp.index.astype(int)

    cmap = plt.get_cmap("crest")

    # Normalize your data using logarithmic normalization
    norm = mcolors.Normalize(vmin=tmp.columns.min(), vmax=tmp.columns.max())

    for idx, row in tmp.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for idx in df.index:
        plt.plot(ls, curve(idx, ls), color=cmap(norm(idx)), label=idx, linestyle=":")
    plt.xlabel(r"Average Bytes per Image $L$ (KB)", fontsize=12)
    plt.ylabel(r"$Err_\text{test}$ (F1)", fontsize=12)

    plt.legend(
        title="Num Samples",
        loc=(0.53, 0.40),
        labels=[f"{x:,}" for x in df.index],
        fontsize=10,
        title_fontsize=10,
    )
    plt.loglog()
    ticks = [6e3, 1e4, 2e4, 3e4, 4e4]
    plt.xticks(ticks, [int(x / 1e3) for x in ticks], fontsize=12)
    plt.yticks([0.2, 0.3, 0.4, 0.6], ["0.2", "0.3", "0.4", "0.6"], fontsize=12)
    plt.tight_layout()
    plt.show()


def food101_alt_metric_scaling_plots():
    a, b, alpha, beta = [
        6.955628723309485,
        1684.6642068710826,
        0.33683224185776905,
        1.0817046562687787,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta)

    ns = np.linspace(2900, 61000, 100)
    ls = np.linspace(5000, 40000, 100)

    df = get_food101_alt_metric_df()
    food101_alt_metric_scaling_n(df, curve, ns)
    food101_alt_metric_scaling_l(df, curve, ls)


def food101_test_compression_plot():
    results = []
    for i in range(2):
        tmp = pd.read_csv(
            os.path.join(FOOD101_PATH, f"cmprs_first_iter{i}/test_set_compression.csv"),
            index_col=0,
        )
        results.append(tmp)
    results = pd.concat(results)
    results = results.groupby(["train_dist", "test_dist"]).mean().reset_index()
    with open(FOOD101_SIZES, "r") as f:
        sizes = json.load(f)

    results["Avg Train Size (KB)"] = (
        results["train_dist"].astype(str).apply(lambda x: sizes[x] // 1e3).astype(int)
    )
    results["Avg Test Size (KB)"] = (
        results["test_dist"].astype(str).apply(lambda x: sizes[x] // 1e3).astype(int)
    )

    sns.set_style("whitegrid", {"axes.grid": False})
    grid_min = results["Avg Test Size (KB)"].min()
    grid_max = results["Avg Test Size (KB)"].max()
    grid_x, grid_y = np.mgrid[
        grid_min:grid_max:100j, grid_min:grid_max:100j
    ]  # Here, 100j means we want 100 points

    z = results.accuracy
    grid_z = griddata(
        (results["Avg Test Size (KB)"], results["Avg Train Size (KB)"]),
        results.accuracy,
        (grid_x, grid_y),
        method="linear",
    )
    cmap = "crest"
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(
        grid_z.T,
        origin="lower",
        extent=[grid_min, grid_max, grid_min, grid_max],
        cmap=cmap,
    )
    sns.scatterplot(
        data=results,
        x="Avg Test Size (KB)",
        y="Avg Train Size (KB)",
        hue="accuracy",
        s=100,
        palette=cmap,
    )
    sns.despine(left=True, bottom=True)
    margin = 0.7
    plt.xlim(grid_min - margin, grid_max + margin)
    plt.ylim(grid_min - margin, grid_max + margin)
    plt.legend(title="Accuracy", fontsize=11, loc=(0.45, 0.5))
    plt.xlabel("Avg Test Size (KB)", fontsize=13)
    plt.ylabel("Avg Train Size (KB)", fontsize=13)
    plt.xticks(np.arange(5, 36, 5).astype(int))
    plt.yticks(np.arange(5, 36, 5).astype(int))
    plt.tight_layout()
    plt.show()
