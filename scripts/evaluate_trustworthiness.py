import argparse
import os
import h5py

def evaluate_trustworthiness(original_neighbors, reduced_neighbors, N_total, k=100):
    """
    Truncated Trustworthiness measures false neighbors.
    Since we only have top-k indices, false neighbors' true rank r(i,j) is unknown.
    We approximate the missing rank with the statistically expected rank N/2.
    """
    N = original_neighbors.shape[0]
    k = min(k, original_neighbors.shape[1], reduced_neighbors.shape[1])

    # Calculate total overlap to quickly deduce false neighbors
    total_overlap = sum(
        len(set(orig).intersection(red))
        for orig, red in zip(original_neighbors[:, :k], reduced_neighbors[:, :k])
    )
    total_false_neighbors = (N * k) - total_overlap

    expected_rank = N_total / 2.0
    penalty_per_item = max(0, expected_rank - k)
    tk_penalty = total_false_neighbors * penalty_per_item

    tk_denom = N * k * (2 * N_total - 3 * k - 1)
    tk = 1.0 - (2 * tk_penalty / tk_denom)
    return tk

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trustworthiness (T_k).")
    parser.add_argument("original_path", type=str, help="Path to original HDF5 dataset.")
    parser.add_argument("reduced_path", type=str, help="Path to reduced HDF5 dataset.")
    parser.add_argument("--k", type=int, default=100, help="Number of neighbors (default: 100).")
    args = parser.parse_args()

    with h5py.File(args.original_path, "r") as f:
        train_neighbors = f["train_neighbors"][:]
        N_total_train = f["train"].shape[0]
        neighbors = f["neighbors"][:]
        N_total_test = f["test"].shape[0]

    with h5py.File(args.reduced_path, "r") as f:
        if "red_train_neighbors" not in f:
            raise ValueError("red_train_neighbors not found! Please compute using dataset_neighbors_gpu.py first.")
        red_train_neighbors = f["red_train_neighbors"][:]
        if "red_neighbors" not in f:
            raise ValueError("red_neighbors not found! Please compute using dataset_neighbors_gpu.py first.")
        red_neighbors = f["red_neighbors"][:]

    tk_train = evaluate_trustworthiness(train_neighbors, red_train_neighbors, N_total_train, k=args.k)
    tk_test = evaluate_trustworthiness(neighbors, red_neighbors, N_total_test, k=args.k)
    print(f"Trustworthiness train (T_{args.k}): {tk_train:.4f}  (Higher is better, Max: 1.0)")
    print(f"Trustworthiness test  (T_{args.k}): {tk_test:.4f}  (Higher is better, Max: 1.0)")

    # --- Save to CSV ---
    reduced_dir = os.path.dirname(args.reduced_path)
    base_name = os.path.basename(args.reduced_path)
    if base_name.endswith(".hdf5"):
        base_name = base_name[:-5]

    csv_path = os.path.join(reduced_dir, f"{base_name}_trustworthiness_k={args.k}.csv")
    with open(csv_path, "w") as f:
        f.write("train;test\n")
        f.write(f"{tk_train};{tk_test}\n")

if __name__ == "__main__":
    main()
