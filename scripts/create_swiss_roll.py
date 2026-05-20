import os
import argparse
import h5py
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def main():
    parser = argparse.ArgumentParser(description="Create Swiss Roll dataset for dimensionality reduction.")
    parser.add_argument("output_dir", type=str, help="Directory path to save the generated HDF5 dataset.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of sample points to generate.")
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise added to the data.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to reserve as queries (test set).")
    
    args = parser.parse_args()

    print(f"Generating Swiss Roll dataset with {args.n_samples} samples and {args.noise} noise...")
    # Generate Swiss Roll dataset
    X, _ = make_swiss_roll(n_samples=args.n_samples, noise=args.noise, random_state=42)
    
    # Train/Test split
    train, test = train_test_split(X, test_size=args.test_size, random_state=42)
    
    print(f"Train size: {train.shape}, Test size (queries): {test.shape}")
    
    n_jobs = int(os.environ.get('PBS_NCPUS', os.cpu_count()))
    
    # Compute 100 + 1 nearest neighbors for train on train
    print("Computing train neighbors...")
    nbrs_train = NearestNeighbors(n_neighbors=101, n_jobs=n_jobs, metric="euclidean", algorithm="brute").fit(train)
    train_neighbors = nbrs_train.kneighbors(train, return_distance=False)
    train_neighbors = train_neighbors[:, 1:]  # Remove the datapoint itself to leave exactly 100
    
    # Compute 100 nearest neighbors for test on train (queries)
    print("Computing test neighbors (queries)...")
    nbrs_test = NearestNeighbors(n_neighbors=100, n_jobs=n_jobs, metric="euclidean", algorithm="brute").fit(train)
    test_neighbors = nbrs_test.kneighbors(test, return_distance=False)
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_path = os.path.join(args.output_dir, "swiss_roll-euclidean.hdf5")
    print(f"Saving dataset to {file_path}...")
    
    # Save the dataset attributes to the .hdf5 file using the standard structure
    with h5py.File(file_path, "w") as f:
        f.create_dataset("train", data=train, dtype='f4')
        f.create_dataset("test", data=test, dtype='f4')
        f.create_dataset("train_neighbors", data=train_neighbors)
        f.create_dataset("neighbors", data=test_neighbors)
        
    print("Swiss Roll dataset created and saved successfully.")

if __name__ == "__main__":
    main()