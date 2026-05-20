import os
import re
import glob
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── Thesis RC ─────────────────────────────────────────────────────────────────
THESIS_RC = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "normal",
    "axes.labelsize":        10,
    "axes.labelcolor":       "#444444",
    "xtick.labelsize":       8.5,
    "ytick.labelsize":       8.5,
    "xtick.color":           "#666666",
    "ytick.color":           "#666666",
    "axes.linewidth":        0.0,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.color":            "#E0E0E0",
    "grid.linewidth":        0.8,
    "grid.linestyle":        "-",
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      0,
    "ytick.major.size":      0,
    "xtick.major.width":     0,
    "ytick.major.width":     0,
    "xtick.minor.visible":   False,
    "ytick.minor.visible":   False,
    "savefig.dpi":           200,
    "legend.frameon":        False,
    "legend.fontsize":       8.5,
}

CMAP       = "viridis"
POINT_SIZE = 6
ALPHA_2D   = 0.80
ALPHA_3D   = 0.55
DPI        = 200
FIG_SIZE   = (5.2, 4.6)

TECHNIQUE_LABELS = {
    "PCA":   "PCA",
    "TSNE":  "t-SNE",
    "UMAP":  "UMAP",
    "SRP":   "Sparse RP",
    "GRP":   "Gaussian RP",
    "SNNAE": "SNN Autoencoder",
    "AE":    "Autoencoder",
    "ORIG":  "Original",
}

DATASET_LABELS = {
    "swiss_roll":                  "Swiss Roll",
    "s_curve":                     "S-Curve",
    "clusters":                    "Gaussian Clusters",
    "sift-128-euclidean":          "SIFT1M",
    "gist-960-euclidean":          "GIST1M",
    "fashion-mnist-784-euclidean": "Fashion-MNIST",
    "laion-768-cosine":            "LAION-2B 300K",
}


def parse_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    techniques = "|".join(TECHNIQUE_LABELS.keys())
    m = re.search(rf"_({techniques})_(\d+)_", name)
    if m:
        technique   = m.group(1)
        dim         = int(m.group(2))
        dataset_key = name[: m.start()]
    else:
        technique, dim, dataset_key = "Unknown", "?", name
    dataset = DATASET_LABELS.get(dataset_key, dataset_key.replace("-", " ").replace("_", " "))
    return dataset, technique, dim


# Dataset folder names (as used in experiments/)
TARGET_DATASETS = [
    "swiss_roll-euclidean",
    "s_curve-euclidean",
    "clusters-euclidean",
]

QUERY_COLOR = "#e63946"
QUERY_SIZE  = POINT_SIZE + 3
QUERY_ALPHA = 0.90


def _load(path, with_queries):
    with h5py.File(path, "r") as f:
        red_train  = f["red_train"][:]
        colors_all = f["viz_color"][:] if "viz_color" in f else None
        red_test   = f["red_test"][:] if (with_queries and "red_test" in f) else None
    return red_train, colors_all, red_test


def _subsample(arr, colors):
    n = len(arr)
    c = colors if colors is not None else np.arange(n, dtype=float)
    return arr, c, f"{n:,} points"


def _plot_2d(ax, data, colors, test_data=None):
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    ax.scatter(
        data[:, 0], data[:, 1],
        c=colors, cmap=CMAP, norm=norm,
        s=POINT_SIZE, alpha=ALPHA_2D, linewidths=0,
        rasterized=True,
    )
    if test_data is not None:
        ax.scatter(
            test_data[:, 0], test_data[:, 1],
            c=QUERY_COLOR, s=QUERY_SIZE, alpha=QUERY_ALPHA, linewidths=0,
            rasterized=True, zorder=5,
        )
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("x", color="#555555")
    ax.set_ylabel("y", color="#555555")


def _plot_3d(ax, data, colors, test_data=None):
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=colors, cmap=CMAP, norm=norm,
        s=POINT_SIZE, alpha=ALPHA_3D, linewidths=0,
        rasterized=True,
    )
    if test_data is not None:
        ax.scatter(
            test_data[:, 0], test_data[:, 1], test_data[:, 2],
            c=QUERY_COLOR, s=QUERY_SIZE, alpha=QUERY_ALPHA, linewidths=0,
            rasterized=True, zorder=5,
        )
    ax.set_xlabel("x", labelpad=3, fontsize=9, color="#555555")
    ax.set_ylabel("y", labelpad=3, fontsize=9, color="#555555")
    ax.set_zlabel("z", labelpad=3, fontsize=9, color="#555555")
    ax.tick_params(labelsize=7, pad=1, colors="#666666")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#E8E8E8")
    ax.grid(True, linewidth=0.3, color="#E0E0E0")
    ax.view_init(elev=20, azim=80)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def visualize_file(hdf5_path, output_dir, with_queries=False):
    dataset, technique, _ = parse_filename(hdf5_path)
    label = TECHNIQUE_LABELS.get(technique, technique)

    red_train, colors_all, red_test = _load(hdf5_path, with_queries)
    _, actual_dim = red_train.shape

    if actual_dim not in (2, 3):
        print(f"  Skipping {os.path.basename(hdf5_path)}: {actual_dim}D not supported")
        return

    data, colors, train_label = _subsample(red_train, colors_all)

    if with_queries and red_test is not None:
        caption = f"{train_label} + {len(red_test):,} queries"
    else:
        red_test = None
        caption  = train_label

    title = f"{dataset}  —  {label}  ({actual_dim}D)"
    plt.rcParams.update(THESIS_RC)

    if actual_dim == 3:
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_title(title, pad=10, color="#333333")
        _plot_3d(ax, data, colors, red_test)
    else:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        ax.set_title(title, pad=10, color="#333333")
        _plot_2d(ax, data, colors, red_test)

    fig.text(0.5, 0.0, caption, ha="center", va="bottom",
             fontsize=7, color="#888888", transform=fig.transFigure)
    fig.tight_layout()

    stem     = os.path.splitext(os.path.basename(hdf5_path))[0]
    suffix   = "_queries" if with_queries else ""
    out_path = os.path.join(output_dir, f"{stem}{suffix}.pdf")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"  Saved -> {out_path}")
    plt.close(fig)


def process_sample_folder(sample_dir, viz_dir):
    files = sorted(glob.glob(os.path.join(sample_dir, "*.hdf5")))
    if not files:
        print(f"  No .hdf5 files in {sample_dir}")
        return
    for f in files:
        print(f"  {os.path.basename(f)}")
        visualize_file(f, viz_dir, with_queries=False)
        visualize_file(f, viz_dir, with_queries=True)


def visualize_original(hdf5_path, output_dir, with_queries=False):
    """Visualize a raw dataset file with train/test keys."""
    stem    = os.path.splitext(os.path.basename(hdf5_path))[0]
    dataset = DATASET_LABELS.get(stem, stem.replace("-", " ").replace("_", " "))

    with h5py.File(hdf5_path, "r") as f:
        train = f["train"][:]
        test  = f["test"][:] if (with_queries and "test" in f) else None

    _, actual_dim = train.shape
    if actual_dim not in (2, 3):
        print(f"  Skipping {os.path.basename(hdf5_path)}: {actual_dim}D not supported")
        return

    colors  = np.arange(len(train), dtype=float)
    caption = f"{len(train):,} points" + (f" + {len(test):,} queries" if test is not None else "")
    title   = f"{dataset}  —  Original  ({actual_dim}D)"

    plt.rcParams.update(THESIS_RC)

    if actual_dim == 3:
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_title(title, pad=10, color="#333333")
        _plot_3d(ax, train, colors, test)
    else:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
        ax.set_title(title, pad=10, color="#333333")
        _plot_2d(ax, train, colors, test)

    fig.text(0.5, 0.0, caption, ha="center", va="bottom",
             fontsize=7, color="#888888", transform=fig.transFigure)
    fig.tight_layout()

    suffix   = "_queries" if with_queries else ""
    out_path = os.path.join(output_dir, f"{stem}{suffix}.pdf")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"  Saved -> {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-visualize reduced and original datasets.")
    parser.add_argument("experiments_dir",
        help="Root experiments directory (e.g. /storage/.../experiments).")
    args = parser.parse_args()

    exper = args.experiments_dir

    # ── Reduced: dim 1 and 2 subfolders per dataset ───────────────────────────
    for dataset_name in TARGET_DATASETS:
        dataset_dir = os.path.join(exper, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"\nSkipping {dataset_name}: {dataset_dir} does not exist")
            continue
        for sample_folder in ("1", "2"):
            sample_dir = os.path.join(dataset_dir, sample_folder)
            if not os.path.isdir(sample_dir):
                print(f"\nSkipping {dataset_name}/{sample_folder}: does not exist")
                continue
            viz_dir = os.path.join(sample_dir, "visualize")
            print(f"\n=== {dataset_name}/{sample_folder} ===")
            process_sample_folder(sample_dir, viz_dir)

    # ── Original 3D: raw dataset files in datasets/ ───────────────────────────
    datasets_dir = os.path.join(exper, "datasets")
    orig_viz_dir = os.path.join(datasets_dir, "visualize")
    print(f"\n=== Original datasets ({datasets_dir}) ===")
    for dataset_name in TARGET_DATASETS:
        hdf5_path = os.path.join(datasets_dir, f"{dataset_name}.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"  Skipping {dataset_name}.hdf5: does not exist")
            continue
        print(f"  {dataset_name}.hdf5")
        visualize_original(hdf5_path, orig_viz_dir, with_queries=False)
        visualize_original(hdf5_path, orig_viz_dir, with_queries=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
