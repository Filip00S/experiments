import os
import sys
import argparse
import h5py
import numpy as np
import torch

def load_hdf5_data(dataset_path: str):
    with h5py.File(dataset_path, "r") as f:
        red_train = f["red_train"][:]
        red_test = f["red_test"][:]
        return red_train, red_test

def save_hdf5_neighbors(red_train_neighbors, red_neighbors, dataset_path):
    with h5py.File(dataset_path, "a") as f:
        # Replace or create the datasets if they already exist
        if "red_train_neighbors" in f:
            del f["red_train_neighbors"]
        f.create_dataset("red_train_neighbors", data=red_train_neighbors)
        
        if "red_neighbors" in f:
            del f["red_neighbors"]
        f.create_dataset("red_neighbors", data=red_neighbors)

def compute_neighbors_gpu(queries, database, metric="euclidean", k=100, batch_size=1024, is_self=False):
    device = torch.device("cuda")
    
    n_queries = queries.shape[0]
    
    print("Loading datasets onto GPU...")
    query_tensor = torch.tensor(queries, dtype=torch.float32).to(device)
    db_tensor = torch.tensor(database, dtype=torch.float32).to(device)
    
    if metric == "cosine":
        print("Normalizing data for cosine distance calculation...")
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        if not is_self:
            db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
        else:
            db_tensor = query_tensor
        
    neighbors = np.zeros((n_queries, k), dtype=np.int32)
    search_k = k + 1 if is_self else k
    
    print(f"Computing neighbors in batches of {batch_size}...")
    for i in range(0, n_queries, batch_size):
        end = min(i + batch_size, n_queries)
        batch = query_tensor[i:end]
        
        if metric == "cosine":
            # Cosine similarity is matrix multiplication of L2-normalized vectors
            sim = torch.mm(batch, db_tensor.T)
            # Largest=True because higher cosine similarity means closer distance
            _, topk_indices = torch.topk(sim, k=search_k, dim=1, largest=True)
        else:
            # Exact Euclidean distance calculation
            dist = torch.cdist(batch, db_tensor)
            # Largest=False because lower euclidean distance means closer
            _, topk_indices = torch.topk(dist, k=search_k, dim=1, largest=False)
            
        if is_self:
            # Copy batch results back to CPU memory, ignoring the 0th index (the point itself)
            neighbors[i:end] = topk_indices.cpu().numpy()[:, 1:]
        else:
            # Keep all top k elements since queries are different from the database
            neighbors[i:end] = topk_indices.cpu().numpy()
        
        if (i % (batch_size * 10)) < batch_size and i > 0:
            print(f"Progress: {i}/{n_queries} queries processed.")
            
    return neighbors

def main():
    # Immediate early exit if CUDA is unavailable
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! Exiting to prevent slow/partial computation.", file=sys.stderr)
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Compute nearest neighbors for a reduced dataset using GPU (PyTorch).")
    parser.add_argument("dataset_path", type=str, help="Path to the reduced HDF5 dataset file.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for queries to avoid out-of-memory errors.")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Early exit check to avoid recomputing finished datasets and wasting queue time
    with h5py.File(dataset_path, "r") as f:
        if "red_train_neighbors" in f and "red_neighbors" in f:
            print(f"Neighbors already computed for '{dataset_path}'. Skipping.")
            sys.exit(0)
            
    red_train, red_test = load_hdf5_data(dataset_path)

    metric = "cosine" if "cosine" in dataset_path.lower() else "euclidean"
    print(f"Computing {metric} neighbors for {dataset_path}...")

    print("\n[1/2] Computing red_train vs red_train (red_train_neighbors)...")
    red_train_neighbors = compute_neighbors_gpu(red_train, red_train, metric=metric, batch_size=args.batch_size, is_self=True)

    print("\n[2/2] Computing red_test vs red_train (red_neighbors)...")
    red_neighbors = compute_neighbors_gpu(red_test, red_train, metric=metric, batch_size=args.batch_size, is_self=False)

    save_hdf5_neighbors(red_train_neighbors, red_neighbors, dataset_path)
    print("Neighbors saved successfully.")

if __name__ == "__main__":
    main()