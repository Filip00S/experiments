import argparse
import os
import h5py
from sklearn.neighbors import NearestNeighbors

def main():
    parser = argparse.ArgumentParser(description="Merge train and test HDF5 into one file.")
    parser.add_argument("train_file", type=str, help="Path to base data.")
    parser.add_argument("test_file", type=str, help="Path to query data.")
    parser.add_argument("output_file", type=str, help="Path to output file.")
    
    args = parser.parse_args()

    print(f"Opening train file: {args.train_file}")
    print(f"Opening test file: {args.test_file}")
    
    with h5py.File(args.train_file, "r") as f_train, \
         h5py.File(args.test_file, "r") as f_test, \
         h5py.File(args.output_file, "w") as f_out:
        
        # Safely grab the first dataset key (often 'emb' in SISAP challenge files)
        train_key = "emb" if "emb" in f_train else list(f_train.keys())[0]
        test_key = "emb" if "emb" in f_test else list(f_test.keys())[0]
        
        print("Extracting train and test datasets...")
        train_data = f_train[train_key][:]
        test_data = f_test[test_key][:]
        
        print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        
        print("Computing 100 nearest neighbors for test queries in train dataset using brute-force cosine distance...")
        n_jobs = int(os.environ.get('PBS_NCPUS', os.cpu_count()))
        nbrs = NearestNeighbors(n_neighbors=100, algorithm="brute", metric="cosine", n_jobs=n_jobs)
        nbrs.fit(train_data)
        test_neighbors = nbrs.kneighbors(test_data, return_distance=False)
        
        print(f"Saving merged dataset to {args.output_file}...")
        f_out.create_dataset("train", data=train_data, dtype='f4')
        f_out.create_dataset("test", data=test_data, dtype='f4')
        f_out.create_dataset("neighbors", data=test_neighbors)
        
    print("Merge complete!")

if __name__ == "__main__":
    main()