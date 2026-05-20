import argparse
import os
import h5py

def evaluate_np(original_neighbors, reduced_neighbors, k=100):
    N = original_neighbors.shape[0]
    k = min(k, original_neighbors.shape[1], reduced_neighbors.shape[1])

    # Fast list comprehension for set intersections
    overlap_sum = sum(
        len(set(orig).intersection(red))
        for orig, red in zip(original_neighbors[:, :k], reduced_neighbors[:, :k])
    )

    return overlap_sum / (N * k)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Neighborhood Precision (NP_k).")
    parser.add_argument("original_path", type=str, help="Path to original HDF5 dataset.")
    parser.add_argument("reduced_path", type=str, help="Path to reduced HDF5 dataset.")
    parser.add_argument("--k", type=int, default=100, help="Number of neighbors (default: 100).")
    args = parser.parse_args()

    with h5py.File(args.original_path, "r") as f:
        train_neighbors = f["train_neighbors"][:]
        neighbors = f["neighbors"][:]

    with h5py.File(args.reduced_path, "r") as f:
        if "red_train_neighbors" not in f:
            raise ValueError("red_train_neighbors not found! Please compute using dataset_neighbors_gpu.py first.")
        red_train_neighbors = f["red_train_neighbors"][:]
        if "red_neighbors" not in f:
            raise ValueError("red_neighbors not found! Please compute using dataset_neighbors_gpu.py first.")
        red_neighbors = f["red_neighbors"][:]

    print(f"Evaluating Neighborhood Precision for top-{args.k} neighbors...")

    npk_train = evaluate_np(train_neighbors, red_train_neighbors, k=args.k)
    npk_test = evaluate_np(neighbors, red_neighbors, k=args.k)

    print(f"Neighborhood Precision train (NP_{args.k}): {npk_train:.4f}  (Higher is better, Max: 1.0)")
    print(f"Neighborhood Precision test  (NP_{args.k}): {npk_test:.4f}  (Higher is better, Max: 1.0)")

    # --- Save to CSV ---
    reduced_dir = os.path.dirname(args.reduced_path)
    base_name = os.path.basename(args.reduced_path)
    if base_name.endswith(".hdf5"):
        base_name = base_name[:-5]

    csv_path = os.path.join(reduced_dir, f"{base_name}_np_k={args.k}.csv")
    with open(csv_path, "w") as f:
        f.write("train;test\n")
        f.write(f"{npk_train};{npk_test}\n")

if __name__ == "__main__":
    main()
