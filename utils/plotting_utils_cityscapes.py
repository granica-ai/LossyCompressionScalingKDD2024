import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import simplejson as json
import glob
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata

from utils.consts import CITYSCAPES_PATH, CITYSCAPES_SIZES, CITYSCAPES_SS


def get_cityscapes_df():
    path_fmt = os.path.join(
        CITYSCAPES_PATH, "dist{dist}_frac{frac}/*/vis_data/scalars.json"
    )

    dists = [1, 2, 4, 7, 10, 15]
    fracs = [f"{x:.1f}" for x in np.arange(0.2, 1.1, 0.2)]
    miou = pd.DataFrame(index=dists, columns=fracs)

    for dist in dists:
        for frac in fracs:
            paths = glob.glob(path_fmt.format(dist=dist, frac=frac))
            if len(paths) == 0:
                continue
            path = paths[-1]
            with open(path, "r") as f:
                data = f.readlines()

            results = {}
            idx = -1
            try:
                while "mIoU" not in results:
                    results = eval(data[idx])
                    idx -= 1
            except IndexError:
                print("No mIoU found in {}".format(path))
                continue

            miou.at[dist, frac] = results["mIoU"]

    miou.rename(
        columns=dict(zip(miou.columns, [float(x) for x in miou.columns])), inplace=True
    )
    miou = (1 - miou / 100).rename(
        columns=dict(
            zip(
                [0.2, 0.4, 0.6, 0.8, 1.0],
                [int(2975 * x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]],
            )
        )
    )
    with open(
        CITYSCAPES_SIZES,
        "r",
    ) as f:
        dist2pixels = json.load(f)
    dist2pixels = {int(k): v for k, v in dist2pixels.items()}
    miou.index = [int(dist2pixels[x]) for x in miou.index]
    return miou.sort_index()


def cityscapes_scaling_n(df, curve, ns):
    df_t = df.T
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")
    norm = mcolors.LogNorm(vmin=df_t.columns.min(), vmax=df_t.columns.max())

    for idx, row in df_t.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in df_t.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$")
    plt.ylabel(r"Test Error $Err_\text{test}$ (mIoU)")
    plt.legend(
        title="Avg Bytes / Img",
        loc="upper right",
        labels=[f"{x:,}" for x in df_t.columns],
        fontsize=8,
        title_fontsize=10,
    )
    plt.loglog()
    plt.yticks([0.18, 0.2, 0.22, 0.24, 0.26, 0.28], [0.18, 0.2, 0.22, 0.24, 0.26, 0.28])
    plt.tight_layout()
    plt.show()


def cityscapes_scaling_l(tmp, curve, ls):
    plt.figure(figsize=(4.5, 3.5))
    tmp.index = tmp.index.astype(int)

    cmap = plt.get_cmap("crest")
    norm = mcolors.Normalize(vmin=tmp.columns.min(), vmax=tmp.columns.max())

    for idx, row in tmp.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for idx in tmp.columns:
        plt.plot(ls, curve(idx, ls), color=cmap(norm(idx)), label=idx, linestyle=":")
    plt.xlabel(r"Average Bytes per Image $L$")
    plt.ylabel(r"Test Error $Err_\text{test}$ (mIoU)")
    plt.legend(
        title="Num Samples",
        loc="upper right",
        labels=[f"{x:,}" for x in tmp.columns],
        fontsize=8,
        title_fontsize=10,
    )
    plt.loglog()
    plt.yticks([0.18, 0.2, 0.22, 0.24, 0.26, 0.28], [0.18, 0.2, 0.22, 0.24, 0.26, 0.28])
    plt.xticks(
        [30000, 40000, 60000, 100000, 200000],
        [f"{int(x/1e3)}k" for x in [30000, 40000, 60000, 100000, 200000]],
    )
    plt.tight_layout()
    plt.show()


def cityscapes_scaling_plots():
    a, b, alpha, beta, c = [
        4.286691211158588,
        428277.4784308216,
        0.58938638522845,
        1.6064150537051824,
        0.13548027044911312,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta) + c

    ns = np.linspace(500, 3000, 100)
    ls = np.linspace(28000, 190000, 100)

    df = get_cityscapes_df()
    cityscapes_scaling_n(df, curve, ns)
    cityscapes_scaling_l(df, curve, ls)


def cityscapes_opt_n(opt_n, uncompressed_n):
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        CITYSCAPES_SS,
        opt_n,
        label="Scaling Curve Opt",
        marker="o",
    )
    plt.plot(
        CITYSCAPES_SS,
        uncompressed_n,
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Number of Samples $n$", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks([4e7, 6e7, 1e8, 1.2e8], ["40", "60", "100", "120"], fontsize=12)
    plt.yticks([20, 100, 300, 1e3, 3e3], [20, 100, 300, "1000", "3000"], fontsize=12)
    plt.tight_layout()
    plt.show()


def cityscapes_opt_l(opt_n, uncompressed_n):
    avg_l = np.divide(CITYSCAPES_SS, opt_n)
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        CITYSCAPES_SS,
        avg_l,
        label="Scaling Curve Opt",
        marker="o",
    )
    unc_avg_l = np.divide(CITYSCAPES_SS, uncompressed_n)
    plt.plot(
        CITYSCAPES_SS,
        unc_avg_l,
        label="Original Format",
        marker="o",
    )
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"Avg Bytes per Image $L$ (KB)", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks([4e7, 6e7, 1e8, 1.2e8], ["40", "60", "100", "120"], fontsize=12)
    plt.yticks([3e4, 1e5, 3e5, 1e6, 2e6], [30, 100, 300, "1000", "2000"], fontsize=12)
    plt.tight_layout()
    plt.show()


def cityscapes_opt_n_l_plots():
    files = glob.glob(
        os.path.join(CITYSCAPES_PATH, "uncompressed", "uncompressed_s[0-9]*.txt")
    )
    files = [x for x in files if "seed" not in x]
    ss = sorted([int(x.split("_s")[-1].split(".")[0]) for x in files])
    uncompressed_n = []
    for s in ss:
        f = os.path.join(CITYSCAPES_PATH, "uncompressed", f"uncompressed_s{s}.txt")
        with open(f, "r") as f:
            uncompressed_n.append(sum(1 for line in f))

    opt_n = []
    for s in ss:
        f = glob.glob(
            os.path.join(CITYSCAPES_PATH, "deterministic", f"idxs_s{s}_n[1-9]*.npy")
        )

        with open(f[0], "rb") as f:
            opt_n.append(sum(1 for line in f))

    cityscapes_opt_n(opt_n, uncompressed_n)
    cityscapes_opt_l(opt_n, uncompressed_n)


def get_cityscapes_results(file_list):

    miou = np.full(len(file_list), np.nan)
    for i, file in enumerate(file_list):
        paths = glob.glob(file + "*/vis_data/scalars.json")
        if len(paths) == 0:
            continue
        path = sorted(paths)[-1]
        # print(path)
        with open(path, "r") as f:
            data = f.readlines()

        results = {}
        idx = -1
        try:
            while "mIoU" not in results:
                results = eval(data[idx])
                idx -= 1
        except IndexError:
            print("No mIoU found in {}".format(path))
            continue

        miou[i] = results["mIoU"]

    return 1 - miou / 100


def get_opt_data():
    model_folders = glob.glob(os.path.join(CITYSCAPES_PATH, "deterministic/s*/"))
    folders2 = glob.glob(os.path.join(CITYSCAPES_PATH, "deterministic/s*_seed0/"))
    folders3 = glob.glob(os.path.join(CITYSCAPES_PATH, "deterministic/s*_seed1/"))
    model_folders = [
        x for x in model_folders if x not in folders2 and x not in folders3
    ]
    ss = [int(x.split("/")[-2][1:-1]) for x in model_folders]
    deterministic1 = get_cityscapes_results(model_folders)
    idx_order = np.argsort(ss)
    deterministic1 = deterministic1[idx_order]

    deterministic2 = get_cityscapes_results(folders2)
    ss = [int(x.split("/")[-2][1:-6]) for x in folders2]
    deterministic2 = deterministic2[np.argsort(ss)]

    deterministic3 = get_cityscapes_results(folders3)
    ss = [int(x.split("/")[-2][1:-6]) for x in folders3]
    deterministic3 = deterministic3[np.argsort(ss)]

    deterministic = np.nanmean(
        np.vstack([deterministic1, deterministic2, deterministic3]), axis=0
    )
    return deterministic


def get_uncompressed_data():
    folders1 = glob.glob(os.path.join(CITYSCAPES_PATH, "uncompressed", "s*/"))
    folders2 = glob.glob(os.path.join(CITYSCAPES_PATH, "uncompressed", "s*_seed0/"))
    folders3 = glob.glob(os.path.join(CITYSCAPES_PATH, "uncompressed", "s*_seed1/"))
    folders1 = list(set(folders1) - set(folders2) - set(folders3))
    ss = [int(x.split("/")[-2][1:-1]) for x in folders1]
    uncompressed1 = get_cityscapes_results(folders1)
    uncompressed1 = uncompressed1[np.argsort(ss)]

    ss = [int(x.split("/")[-2][1:-6]) for x in folders2]
    uncompressed2 = get_cityscapes_results(folders2)
    uncompressed2 = uncompressed2[np.argsort(ss)]

    ss = [int(x.split("/")[-2][1:-6]) for x in folders3]
    uncompressed3 = get_cityscapes_results(folders3)
    uncompressed3 = uncompressed3[np.argsort(ss)]
    uncompressed3 = np.concatenate([uncompressed3, [np.nan] * (5 - len(uncompressed3))])

    uncompressed = np.nanmean(
        np.vstack([uncompressed1, uncompressed2, uncompressed3]), axis=0
    )
    return uncompressed


def get_randomized_data():
    folders1 = glob.glob(os.path.join(CITYSCAPES_PATH, "randomized", "s[0-9]*/"))
    folders2 = glob.glob(os.path.join(CITYSCAPES_PATH, "randomized", "s[0-9]*_seed0/"))
    folders3 = glob.glob(os.path.join(CITYSCAPES_PATH, "randomized", "s[0-9]*_seed1/"))
    folders_max1 = glob.glob(
        os.path.join(CITYSCAPES_PATH, "randomized", "s[0-9]*_seed0_max12/")
    )
    folders_max2 = glob.glob(
        os.path.join(CITYSCAPES_PATH, "randomized", "s[0-9]*_seed1_max12/")
    )
    folders1 = list(
        set(folders1)
        - set(folders2)
        - set(folders3)
        - set(folders_max1)
        - set(folders_max2)
    )
    folders2 = [x for x in folders2 if "max" not in x]
    ss = [int(x.split("/")[-2][1:-1]) for x in folders1]
    randomized1 = get_cityscapes_results(folders1)
    randomized1 = randomized1[np.argsort(ss)]

    ss = [int(x.split("/")[-2][1:-6]) for x in folders2 if "max" not in x]
    randomized2 = get_cityscapes_results(folders2)
    randomized2 = randomized2[np.argsort(ss)]

    ss = [int(x.split("/")[-2][1:-6]) for x in folders3 if "max" not in x]
    randomized3 = get_cityscapes_results(folders3)
    randomized3 = randomized3[np.argsort(ss)]

    randomized = np.nanmean(np.vstack((randomized1, randomized2, randomized3)), axis=0)
    return randomized


def cityscapes_opt_s_curve():
    deterministic = get_opt_data()
    uncompressed = get_uncompressed_data()
    randomized = get_randomized_data()
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(
        sorted(CITYSCAPES_SS),
        deterministic,
        label="Scaling Curve Opt",
        marker="o",
        zorder=10,
    )
    plt.plot(sorted(CITYSCAPES_SS), uncompressed, label="Original Format", marker="o")
    plt.plot(sorted(CITYSCAPES_SS), randomized, label="Randomized", marker="o")
    plt.xlabel(r"Storage Size $s$ (MB)", fontsize=12)
    plt.ylabel(r"$Err_\text{test}$ (mIoU)", fontsize=12)
    plt.legend()
    plt.loglog()
    plt.xticks([4e7, 6e7, 1e8, 1.2e8], ["40", "60", "100", "120"], fontsize=12)
    plt.yticks([0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.tight_layout()
    plt.show()


def get_cityscapes_alt_metric_df():
    path_fmt = os.path.join(
        CITYSCAPES_PATH, "dist{dist}_frac{frac}/*/vis_data/scalars.json"
    )

    dists = [1, 2, 4, 7, 10, 15]
    fracs = [f"{x:.1f}" for x in np.arange(0.2, 1.1, 0.2)]
    macc = pd.DataFrame(index=dists, columns=fracs)
    for dist in dists:
        for frac in fracs:
            paths = glob.glob(path_fmt.format(dist=dist, frac=frac))
            if len(paths) == 0:
                continue
            path = paths[-1]
            with open(path, "r") as f:
                data = f.readlines()

            results = {}
            idx = -1
            try:
                while "mIoU" not in results:
                    results = eval(data[idx])
                    idx -= 1
            except IndexError:
                print("No mIoU found in {}".format(path))
                continue

            macc.at[dist, frac] = results["mAcc"]
    macc.rename(
        columns=dict(zip(macc.columns, [float(x) for x in macc.columns])), inplace=True
    )
    with open(
        CITYSCAPES_SIZES,
        "r",
    ) as f:
        dist2pixels = json.load(f)
    dist2pixels = {int(k): v for k, v in dist2pixels.items()}
    macc.index = [int(dist2pixels[x]) for x in macc.index]
    tmp = (1 - macc / 100).rename(
        columns=dict(
            zip(
                [0.2, 0.4, 0.6, 0.8, 1.0],
                [int(2975 * x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]],
            )
        )
    )

    return tmp


def cityscapes_alt_metric_scaling_n(tmp, curve, ns):
    df = tmp.T
    plt.figure(figsize=(4.5, 3.5))

    cmap = plt.get_cmap("crest")
    norm = mcolors.LogNorm(vmin=df.columns.min(), vmax=df.columns.max())

    for idx, row in df.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for col in df.columns:
        plt.plot(ns, curve(ns, col), color=cmap(norm(col)), label=col, linestyle=":")
    plt.xlabel(r"Number of Samples $n$", fontsize=12)
    plt.ylabel(r"$Err_\text{test}$ (mAcc)", fontsize=12)
    plt.legend(
        title="Avg Bytes / Img",
        loc="lower left",
        labels=[f"{x:,}" for x in df.columns],
        fontsize=10,
        title_fontsize=10,
    )
    plt.yticks([])
    plt.loglog()
    plt.yticks(
        [0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        [0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        fontsize=12,
    )
    plt.xticks([600, 1000, 2000, 3000], [600, 1000, 2000, 3000], fontsize=12)
    plt.tight_layout()
    plt.show()


def cityscapes_alt_metric_scaling_l(tmp, curve, ls):
    plt.figure(figsize=(4.5, 3.5))
    tmp.index = tmp.index.astype(int)

    cmap = plt.get_cmap("crest")
    norm = mcolors.Normalize(vmin=tmp.columns.min(), vmax=tmp.columns.max())

    for idx, row in tmp.sort_index().T.iterrows():
        plt.scatter(row.index, row.values, color=cmap(norm(row.name)))
    for idx in tmp.columns:
        plt.plot(ls, curve(idx, ls), color=cmap(norm(idx)), label=idx, linestyle=":")
    plt.xlabel(r"Average Bytes per Image $L$ (KB)", fontsize=12)
    plt.ylabel(r"$Err_\text{test}$ (mAcc)", fontsize=12)
    plt.legend(
        title="Num Samples",
        loc=(0.59, 0.47),
        labels=[f"{x:,}" for x in tmp.columns],
        fontsize=10,
        title_fontsize=10,
    )
    plt.loglog()
    plt.yticks(
        [0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        [0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
        fontsize=12,
    )
    ticks = [3e4, 4e4, 6e4, 1e5, 2e5]
    plt.xticks(ticks, [int(x / 1e3) for x in ticks], fontsize=12)
    plt.tight_layout()
    plt.show()


def cityscapes_alt_metric_scaling_plots():
    b, a, beta, alpha, c = [
        23863.367788397576,
        13.085480532675318,
        1.3515628985316512,
        0.7953246891872171,
        0.08242776933985616,
    ]

    def curve(n, L):
        return a * (n**-alpha) + b * (L**-beta) + c

    ns = np.linspace(500, 3000, 100)
    ls = np.linspace(28000, 190000, 100)

    tmp = get_cityscapes_alt_metric_df()
    cityscapes_alt_metric_scaling_n(tmp, curve, ns)
    cityscapes_alt_metric_scaling_l(tmp, curve, ls)


def cityscapes_test_compression_plot():
    dists = [1, 2, 4, 7, 10, 15]
    results = []
    for train_dist in dists:
        for test_dist in dists:
            files = glob.glob(
                os.path.join(
                    CITYSCAPES_PATH,
                    f"dist{train_dist}_frac1.0/test/test_dist{test_dist}/*/*.json",
                )
            )
            if len(files) != 1:
                print(f"Skipping {train_dist} {test_dist} - multiple files found")
            with open(files[0], "r") as f:
                metrics = json.load(f)
            results.append((train_dist, test_dist, metrics["mIoU"] / 100))
    results = pd.DataFrame(results, columns=["train_dist", "test_dist", "mIoU"])

    with open(
        CITYSCAPES_SIZES,
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
    grid_x, grid_y = np.mgrid[
        grid_min:grid_max:100j, grid_min:grid_max:100j
    ]  # Here, 100j means we want 100 points

    grid_z = griddata(
        (results["Avg Test Size (KB)"], results["Avg Train Size (KB)"]),
        results.mIoU,
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
        hue="mIoU",
        s=100,
        palette=cmap,
    )
    sns.despine(left=True, bottom=True)
    margin = 3
    plt.xlim(grid_min - margin, grid_max + margin)
    plt.ylim(grid_min - margin, grid_max + margin)
    plt.legend(title="mIoU", fontsize=11, loc=(0.65, 0.55))
    plt.xlabel("Avg Test Size (KB)", fontsize=13)
    plt.ylabel("Avg Train Size (KB)", fontsize=13)
    plt.xticks(np.arange(25, 180, 25).astype(int))
    plt.yticks(np.arange(25, 180, 25).astype(int))
    plt.tight_layout()
    plt.show()
