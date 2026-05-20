import os
import sys
import argparse

import h5py
from sklearn.neighbors import NearestNeighbors


def load_hdf5_data(dataset_path: str):
    path = dataset_path
    with h5py.File(path, "r") as f:
        data = f["train"][:]

        return data


def save_hdf5_neighbors(neighbors, dataset_path):
    path = dataset_path
    with h5py.File(path, "a") as f:
        f.create_dataset("train_neighbors", data=neighbors)


def main():
    parser = argparse.ArgumentParser(description="Compute nearest neighbors for a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the HDF5 dataset file.")
    args = parser.parse_args()

    # Load file
    dataset_path = args.dataset_path
    data = load_hdf5_data(dataset_path)

    # Auto-detect metric: use cosine if "cosine" is in the dataset name, euclidean for everything else
    metric = "cosine" if "cosine" in dataset_path.lower() else "euclidean"
    print(f"Computing neighbors for {dataset_path} using {metric} distance...")

    n_jobs = int(os.environ.get('PBS_NCPUS', os.cpu_count()))
    # Compute 100 + 1 nearest neighbors of each dataset on itself (+ 1 is the datapoint itself, since it has distance 0)
    nbrs = NearestNeighbors(
        n_neighbors=101, n_jobs=n_jobs, metric=metric, algorithm="brute"
    ).fit(data)
    neighbors = nbrs.kneighbors(data, return_distance=False)

    # Remove itself (+ 1) from list
    neighbors = neighbors[:, 1:]

    # Save file
    save_hdf5_neighbors(neighbors, dataset_path)


if __name__ == "__main__":
    main()
