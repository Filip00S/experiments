import os
import sys
import argparse
import h5py
import numpy as np
import torch

def load_hdf5_data(dataset_path: str):
    with h5py.File(dataset_path, "r") as f:
        train_data = f["train"][:]
        return train_data

def save_hdf5_neighbors(train_neighbors, dataset_path):
    with h5py.File(dataset_path, "a") as f:
        # Replace or create the dataset if it already exists
        if "train_neighbors" in f:
            del f["train_neighbors"]
        f.create_dataset("train_neighbors", data=train_neighbors)

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
            sim = torch.mm(batch, db_tensor.T)
            _, topk_indices = torch.topk(sim, k=search_k, dim=1, largest=True)
        else:
            dist = torch.cdist(batch, db_tensor)
            _, topk_indices = torch.topk(dist, k=search_k, dim=1, largest=False)
            
        if is_self:
            neighbors[i:end] = topk_indices.cpu().numpy()[:, 1:]
        else:
            neighbors[i:end] = topk_indices.cpu().numpy()
        
        if (i % (batch_size * 10)) < batch_size and i > 0:
            print(f"Progress: {i}/{n_queries} queries processed.")
            
    return neighbors

def main():
    # Immediate early exit if CUDA is unavailable
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! Exiting to prevent slow/partial computation.", file=sys.stderr)
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Compute nearest neighbors for an original dataset using GPU (PyTorch).")
    parser.add_argument("dataset_path", type=str, help="Path to the original HDF5 dataset file.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for queries.")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Early exit check to avoid recomputing finished datasets
    with h5py.File(dataset_path, "r") as f:
        if "train_neighbors" in f:
            print(f"Neighbors already computed for '{dataset_path}'. Skipping.")
            sys.exit(0)
            
    train_data = load_hdf5_data(dataset_path)

    metric = "cosine" if "cosine" in dataset_path.lower() else "euclidean"
    print(f"Computing {metric} neighbors for {dataset_path}...")

    print("\nComputing train vs train (train_neighbors)...")
    train_neighbors = compute_neighbors_gpu(train_data, train_data, metric=metric, batch_size=args.batch_size, is_self=True)

    save_hdf5_neighbors(train_neighbors, dataset_path)
    print("Neighbors saved successfully.")

if __name__ == "__main__":
    main()