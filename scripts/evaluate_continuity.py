import argparse
import os
import h5py

def evaluate_continuity(original_neighbors, reduced_neighbors, N_total, k=100):
    """
    Truncated Continuity measures lost true neighbors.
    Since we only have top-k indices, lost neighbors' new rank r'(i,j) is unknown.
    We approximate the missing rank with the statistically expected rank N/2.
    """
    N = original_neighbors.shape[0]
    k = min(k, original_neighbors.shape[1], reduced_neighbors.shape[1])

    # Calculate total overlap to quickly deduce lost true neighbors
    total_overlap = sum(
        len(set(orig).intersection(red))
        for orig, red in zip(original_neighbors[:, :k], reduced_neighbors[:, :k])
    )
    total_missing_neighbors = (N * k) - total_overlap

    expected_rank = N_total / 2.0
    penalty_per_item = max(0, expected_rank - k)
    ck_penalty = total_missing_neighbors * penalty_per_item

    ck_denom = N * k * (2 * N_total - 3 * k - 1)
    ck = 1.0 - (2 * ck_penalty / ck_denom)
    return ck

def main():
    parser = argparse.ArgumentParser(description="Evaluate Continuity (C_k).")
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

    ck_train = evaluate_continuity(train_neighbors, red_train_neighbors, N_total_train, k=args.k)
    ck_test = evaluate_continuity(neighbors, red_neighbors, N_total_test, k=args.k)
    print(f"Continuity train (C_{args.k}): {ck_train:.4f}  (Higher is better, Max: 1.0)")
    print(f"Continuity test  (C_{args.k}): {ck_test:.4f}  (Higher is better, Max: 1.0)")

    # --- Save to CSV ---
    reduced_dir = os.path.dirname(args.reduced_path)
    base_name = os.path.basename(args.reduced_path)
    if base_name.endswith(".hdf5"):
        base_name = base_name[:-5]

    csv_path = os.path.join(reduced_dir, f"{base_name}_continuity_k={args.k}.csv")
    with open(csv_path, "w") as f:
        f.write("train;test\n")
        f.write(f"{ck_train};{ck_test}\n")

if __name__ == "__main__":
    main()
