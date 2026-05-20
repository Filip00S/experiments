import argparse
import os
import h5py

def evaluate_mrre(original_neighbors, reduced_neighbors, N_total, k=100):
    """
    Truncated Mean Relative Rank Error.
    Measures relative shifts in ranks within the top-k structure.
    For true neighbors that completely fall out of the top-k, we penalize
    them heavily by assigning the statistically expected rank N/2.
    """
    N = original_neighbors.shape[0]
    k = min(k, original_neighbors.shape[1], reduced_neighbors.shape[1])

    mrre_sum = 0.0
    expected_rank = N_total / 2.0

    for orig, red in zip(original_neighbors[:, :k], reduced_neighbors[:, :k]):
        # Fast lookup dictionary for {point_id: reduced_rank}
        red_ranks = {val: rank for rank, val in enumerate(red, start=1)}

        for orig_rank, val in enumerate(orig, start=1):
            # If preserved, find relative error. If lost, assign the heavy N/2 penalty.
            new_rank = red_ranks.get(val, expected_rank)

            mrre_sum += abs(orig_rank - new_rank) / orig_rank

    return mrre_sum / (N * k)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mean Relative Rank Error (MRRE_k).")
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

    print(f"Evaluating MRRE for top-{args.k} neighbors (this takes a few seconds)...")
    mrre_train = evaluate_mrre(train_neighbors, red_train_neighbors, N_total_train, k=args.k)
    mrre_test = evaluate_mrre(neighbors, red_neighbors, N_total_test, k=args.k)
    print(f"Mean Relative Rank Error train (MRRE_{args.k}): {mrre_train:.4f}  (Lower is better, Min: 0.0)")
    print(f"Mean Relative Rank Error test  (MRRE_{args.k}): {mrre_test:.4f}  (Lower is better, Min: 0.0)")

    # --- Save to CSV ---
    reduced_dir = os.path.dirname(args.reduced_path)
    base_name = os.path.basename(args.reduced_path)
    if base_name.endswith(".hdf5"):
        base_name = base_name[:-5]

    csv_path = os.path.join(reduced_dir, f"{base_name}_mrre_k={args.k}.csv")
    with open(csv_path, "w") as f:
        f.write("train;test\n")
        f.write(f"{mrre_train};{mrre_test}\n")

if __name__ == "__main__":
    main()
