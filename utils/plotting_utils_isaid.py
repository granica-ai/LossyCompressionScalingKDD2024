import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import simplejson as json
import re
import pandas as pd
import glob
import os
import io
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata

from utils.consts import ISAID_PATH, ISAID_SIZES


def get_last_line_acc(file_name):
    with open(file_name, "rb") as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b"\n":
            f.seek(-2, io.SEEK_CUR)
        last_line = f.readline().decode()
    return last_line


def get_last_line_bbox_map(file_name, metric="bbox_mAP"):
    last_line = get_last_line_acc(file_name)
    match = re.search(f"coco/{metric}" + r": (\d+\.\d+)", last_line)
    if not match:
        return np.nan
    return float(match.group(1))


def get_single_metric_df(dists=None, fracs=None, metric="bbox_mAP"):
    if dists is None:
        dists = [1, 2, 4, 7, 10, 15]
    if fracs is None:
        fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    with open(ISAID_SIZES, "r") as f:
        dist2l = json.load(f)
    results = []
    for dist in dists:
        for frac in fracs:
            files = glob.glob(
                os.path.join(ISAID_PATH, f"dist{dist}_frac{frac}/*/*.log")
            )
            if len(files) == 0:
                continue

            file = sorted(files)[-1]
            bbox_map = get_last_line_bbox_map(file, metric=metric)
            results.append((dist, frac, bbox_map))

    df = pd.DataFrame(results, columns=["dist", "frac", "bbox_map"])
    df = df.pivot(columns="dist", index="frac", values="bbox_map")
    df.columns = [int(dist2l[str(col)]) for col in df.columns]
    return 1 - df


def get_mean_results():
    bbox_metrics = [
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "bbox_mAP_s",
        "bbox_mAP_m",
        "bbox_mAP_l",
    ]
    segm_metrics = [
        "segm_mAP",
        "segm_mAP_50",
        "segm_mAP_75",
        "segm_mAP_s",
        "segm_mAP_m",
        "segm_mAP_l",
    ]

    results = []
    for metric in bbox_metrics + segm_metrics:
        df = get_single_metric_df(metric=metric, dists=[1, 2, 4, 7, 10, 15])
        df = df.stack().reset_index()
        df.columns = ["frac", "L", "metric_value"]
        df["metric"] = metric
        results.append(df)

    results = pd.concat(results)
    lat = pd.read_csv("/mnt/enigma0/km/results/scaling/isaid/latitude_results.csv")
    both_mean = (
        pd.concat([results, lat]).groupby(["metric", "frac", "L"]).mean().reset_index()
    )
    n = 28029
    both_mean["n"] = (n * both_mean["frac"]).astype(int)
    return both_mean


def isaid_scaling_n(df, curve, ns):
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")

    # Normalize your data using logarithmic normalization
    norm = mcolors.LogNorm(vmin=df.columns.min(), vmax=df.columns.max())

    for idx, row in df.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in df.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$", fontsize=12)
    plt.ylabel(r"$Err_{\mathrm{test}}$ (BBox mAP$_s$)", fontsize=12)
    plt.legend(
        title="Avg Bytes / Img",
        loc="lower left",
        labels=[f"{x:,}" for x in df.columns],
        fontsize=9,
        title_fontsize=10,
    )
    plt.loglog()
    plt.xticks(
        [6e3, 8e3, 1e4, 2e4, 3e4],
        [f"{int(x/1e3)}k" for x in [6e3, 8e3, 1e4, 2e4, 3e4]],
        fontsize=12,
    )
    plt.yticks(
        [0.74, 0.76, 0.78, 0.8, 0.82, 0.84],
        [0.74, 0.76, 0.78, 0.80, 0.82, ""],
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()


def isaid_scaling_l(df, curve, ls):
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
    plt.ylabel(r"$Err_\mathrm{test}$ (BBox mAP$_s$)", fontsize=12)

    plt.legend(
        title="Num Samples",
        labels=[f"{x:,}" for x in df.index],
        fontsize=10,
        title_fontsize=10,
    )
    plt.loglog()
    plt.xticks(
        [1e4, 2e4, 4e4, 8e4],
        [f"{int(x/1e3)}" for x in [1e4, 2e4, 4e4, 8e4]],
        fontsize=12,
    )
    plt.yticks(
        [0.74, 0.76, 0.78, 0.8, 0.82, 0.84],
        [0.74, 0.76, 0.78, 0.80, 0.82, ""],
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def isaid_scaling_plots():
    a, b, alpha, beta, c = [
        1.0022450366694422,
        269996.37601695274,
        0.09093488964456031,
        1.6962044400861096,
        0.3331982003692264,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta) + c

    ns = np.linspace(5500, 28100, 100)
    ls = np.linspace(10000, 85000, 100)
    both_mean = get_mean_results()
    df = both_mean.loc[both_mean.metric == "bbox_mAP_s"].pivot(
        index="n", columns="L", values="metric_value"
    )
    isaid_scaling_n(df, curve, ns)
    isaid_scaling_l(df, curve, ls)


def get_s_curve_df(expt_name, metric="bbox_mAP"):
    expts = glob.glob(os.path.join(ISAID_PATH, f"{expt_name}/*/*/*.log"))
    results = []
    for log in expts:
        subexpt = log.split("/")[-3].split("_")
        s = subexpt[0][1:]
        if len(subexpt) == 1:
            seed = 42
        else:
            seed = int(subexpt[1][4:])
        bbox_map = get_last_line_bbox_map(log, metric=metric)
        results.append((s, seed, bbox_map))

    df = pd.DataFrame(results, columns=["s", "seed", metric])
    df["s"] = df["s"].astype(int)
    df[metric] = 1 - df[metric]
    return df


def isaid_opt_s_curve():
    scaling_curve_opt = get_s_curve_df("deterministic", metric="bbox_mAP_s")
    original = get_s_curve_df("uncompressed", metric="bbox_mAP_s")
    randomized = get_s_curve_df("randomized", metric="bbox_mAP_s")

    plt.figure(figsize=(4.5, 3.5))
    scaling_curve_opt.groupby("s").mean().plot(
        y="bbox_mAP_s", marker="o", ax=plt.gca(), label="Scaling Curve Opt", zorder=10
    )
    ax = (
        original.groupby("s")
        .mean()
        .plot(y="bbox_mAP_s", marker="o", ax=plt.gca(), label="Original Format")
    )
    (
        randomized.groupby("s")
        .mean()
        .plot(y="bbox_mAP_s", marker="o", ax=plt.gca(), label="Randomized")
    )

    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"$Err_\mathrm{test}$ (BBox mAP$_s$)", fontsize=12)
    plt.legend(fontsize=10)
    plt.loglog()
    plt.xticks(
        [1e8, 2e8, 3e8, 4e8],
        [f"{int(x/1e6)}" for x in [1e8, 2e8, 3e8, 4e8]],
        fontsize=12,
    )
    plt.yticks([0.7, 0.8, 0.9, 0.95], [0.7, 0.8, 0.9, 0.95], fontsize=12)
    plt.tight_layout()
    plt.show()


def isaid_opt_n(ss, opt_n, uncompressed_n):
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        ss,
        opt_n,
        label="Scaling Curve Opt",
        marker="o",
    )
    plt.plot(
        ss,
        uncompressed_n,
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Number of Samples $n$", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks(
        [1e8, 2e8, 3e8, 4e8],
        [f"{int(x/1e6)}" for x in [1e8, 2e8, 3e8, 4e8]],
        fontsize=12,
    )
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def isaid_opt_l(ss, opt_n, uncompressed_n):
    avg_l = np.divide(ss, opt_n)
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        ss,
        avg_l,
        label="Scaling Curve Opt",
        marker="o",
    )
    unc_avg_l = np.divide(ss, uncompressed_n)
    plt.plot(
        ss,
        unc_avg_l,
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Avg Bytes per Image $L$ (KB)", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks(
        [1e8, 2e8, 3e8, 4e8],
        [f"{int(x/1e6)}" for x in [1e8, 2e8, 3e8, 4e8]],
        fontsize=12,
    )
    ticks = [1e4, 3e4, 1e5, 5e5]
    plt.yticks(ticks, [int(x / 1e3) for x in ticks], fontsize=12)

    plt.tight_layout()
    plt.show()


def isaid_opt_n_l_plots():
    files = glob.glob(
        os.path.join(ISAID_PATH, "uncompressed", "uncompressed_s[0-9]*.json")
    )
    files = [x for x in files if "seed" not in x]
    ss = sorted([int(x.split("_s")[-1].split(".")[0]) for x in files])
    uncompressed_n = []
    for s in ss:
        f = os.path.join(ISAID_PATH, "uncompressed", f"uncompressed_s{s}.json")
        with open(f, "r") as f:
            data = json.load(f)
            uncompressed_n.append(len(data["images"]))

    opt_n = []
    for s in ss:
        f = glob.glob(
            os.path.join(ISAID_PATH, "deterministic", f"idxs_s{s}_n[1-9]*_seed42.json")
        )
        with open(f[0], "r") as f:
            data = json.load(f)
            opt_n.append(len(data["images"]))

    isaid_opt_n(ss, opt_n, uncompressed_n)
    isaid_opt_l(ss, opt_n, uncompressed_n)


def isaid_alt_metric_scaling_n(segm_map, curve, ns):
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")

    # Normalize your data using logarithmic normalization
    norm = mcolors.LogNorm(vmin=segm_map.columns.min(), vmax=segm_map.columns.max())

    for idx, row in segm_map.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in segm_map.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$", fontsize=12)
    plt.ylabel(r"$Err_{\mathrm{test}}$ (Segmentation mAP)", fontsize=12)
    plt.legend(
        title="Avg Bytes / Img",
        loc="lower left",
        labels=[f"{x:,}" for x in segm_map.columns],
        fontsize=9,
        title_fontsize=10,
    )
    plt.loglog()
    plt.xticks(
        [6e3, 8e3, 1e4, 2e4, 3e4],
        [f"{int(x/1e3)}k" for x in [6e3, 8e3, 1e4, 2e4, 3e4]],
        fontsize=12,
    )
    ticks = [0.66, 0.68, 0.7, 0.72, 0.74, 0.76]
    plt.yticks(ticks, ticks, fontsize=12)

    plt.tight_layout()
    plt.show()


def isaid_alt_metric_scaling_l(segm_map, curve, ls):
    # same plot as above, but with continuous colormap
    plt.figure(figsize=(4.5, 3.5))
    tmp = segm_map.T
    tmp.index = tmp.index.astype(int)

    cmap = plt.get_cmap("crest")

    # Normalize your data using logarithmic normalization
    norm = mcolors.Normalize(vmin=tmp.columns.min(), vmax=tmp.columns.max())

    for idx, row in tmp.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for idx in segm_map.index:
        plt.plot(ls, curve(idx, ls), color=cmap(norm(idx)), label=idx, linestyle=":")
    plt.xlabel(r"Average Bytes per Image $L$ (KB)", fontsize=12)
    plt.ylabel(r"$Err_\mathrm{test}$ (Segmentation mAP)", fontsize=12)
    plt.legend(
        title="Num Samples",
        loc=(0.55, 0.47),
        labels=[f"{x:,}" for x in tmp.index],
        fontsize=10,
        title_fontsize=10,
    )

    plt.loglog()
    plt.xticks(
        [1e4, 2e4, 4e4, 8e4],
        [f"{int(x/1e3)}" for x in [1e4, 2e4, 4e4, 8e4]],
        fontsize=12,
    )
    ticks = [0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78]
    plt.yticks(ticks, ticks, fontsize=12)
    plt.tight_layout()
    plt.show()


def isaid_alt_metric_scaling_plots():
    a, b, alpha, beta, c = [
        26.49805947800434,
        327924927011.2764,
        0.6275867996160731,
        3.209303664179312,
        0.610035280351587,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta) + c

    ns = np.linspace(5500, 28100, 100)
    ls = np.linspace(10000, 85000, 100)
    both_mean = get_mean_results()
    segm_map = both_mean.loc[both_mean.metric == "segm_mAP"].pivot(
        index="n", columns="L", values="metric_value"
    )
    isaid_alt_metric_scaling_n(segm_map, curve, ns)
    isaid_alt_metric_scaling_l(segm_map, curve, ls)


def isaid_test_compression_plot():
    dists = [1, 2, 4, 7, 10, 15]
    results = []
    for train_dist in dists:
        for test_dist in dists:
            files = glob.glob(
                os.path.join(
                    ISAID_PATH,
                    f"dist{train_dist}_frac1.0/test/test_dist{test_dist}/*/*.json",
                )
            )
            with open(files[0], "r") as f:
                metrics = json.load(f)
            results.append((train_dist, test_dist, metrics["coco/bbox_mAP_s"]))
    results = pd.DataFrame(
        results, columns=["train_dist", "test_dist", "coco/bbox_mAP_s"]
    )

    with open(
        ISAID_SIZES,
        "r",
    ) as f:
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
    grid_x, grid_y = np.mgrid[grid_min:grid_max:100j, grid_min:grid_max:100j]

    grid_z = griddata(
        (results["Avg Test Size (KB)"], results["Avg Train Size (KB)"]),
        results["coco/bbox_mAP_s"],
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
        hue="coco/bbox_mAP_s",
        s=100,
        palette=cmap,
    )
    sns.despine(left=True, bottom=True)
    margin = 3
    plt.xlim(grid_min - margin, grid_max + margin)
    plt.ylim(grid_min - margin, grid_max + margin)

    plt.xlabel("Avg Test Size (KB)", fontsize=13)
    plt.ylabel("Avg Train Size (KB)", fontsize=13)
    plt.legend(loc=(0.635, 0.48), title=r"BBox mAP$_s$", fontsize=11)
    plt.tight_layout()
    plt.show()
