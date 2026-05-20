import os
import time
import argparse
import h5py

# Limit BLAS/OpenMP threads to allocated CPUs BEFORE importing numpy/sklearn
allocated_cpus = os.environ.get('PBS_NCPUS', str(os.cpu_count()))
os.environ['OMP_NUM_THREADS'] = allocated_cpus
os.environ['OPENBLAS_NUM_THREADS'] = allocated_cpus
os.environ['MKL_NUM_THREADS'] = allocated_cpus

import numpy as np
from sklearn.random_projection import SparseRandomProjection

def reduce_and_save(dataset_path, save_folder, dim, sample_size, mode, rand_state=597):
    
    # --- Early Exit Check ---
    base_dataset_name = os.path.basename(dataset_path)
    if base_dataset_name.endswith(".hdf5"):
        base_dataset_name = base_dataset_name[:-5]
        
    sample_size_str = str(sample_size).replace('.', ',')
    
    technique = "SRP"
    save_filename = f"{base_dataset_name}_{technique}_{dim}_{sample_size_str}.hdf5"
    save_path = os.path.join(save_folder, save_filename)
    
    if os.path.exists(save_path):
        print(f"Output file '{save_path}' already exists. Skipping computation.")
        return

    # --- 0. Setup and Indexing ---
    with h5py.File(dataset_path, "r") as f:
        num_total_samples = f["train"].shape[0]

    # Calculate number of samples for fitting
    num_fit_samples = int(num_total_samples * sample_size)
    
    # --- Optimization: If sample_size is 1.0, we skip creating indices ---
    use_indexing = (sample_size < 1.0)
    
    if use_indexing:
        rng = np.random.RandomState(rand_state)
        fit_indices = rng.choice(num_total_samples, num_fit_samples, replace=False)
        fit_indices.sort() 
    else:
        fit_indices = None # Not used when loading the full dataset

    # --- 1. Load Sampled Data for Fitting ---
    train_sampled = np.array([])
    
    with h5py.File(dataset_path, "r") as f:
        if use_indexing:
            # Load only the necessary sampled data for fitting
            train_sampled = f["train"][fit_indices, :]
        else:
            # Load the whole dataset directly if sample_size is 1.0
            train_sampled = f["train"][:]
    
    # --- 2. Fit SparseRandomProjection ---
    srp_object = SparseRandomProjection(n_components=dim, random_state=rand_state)
    
    start_time = time.time()
    srp_object.fit(train_sampled)
    end_time = time.time()
    fit_time = end_time - start_time

    del train_sampled
    if use_indexing:
        del fit_indices 
    
    # --- 3. Transform Test Data (Query) ---
    test = np.array([])
    with h5py.File(dataset_path, "r") as f:
        test = f["test"][:]

    start_time = time.time()
    red_test = srp_object.transform(test)
    end_time = time.time()
    test_transform_time = end_time - start_time

    del test

    # --- 4. Prepare Output File ---
    with h5py.File(save_path, "w") as f:
        f.create_dataset("red_test", data=red_test, dtype='f4')
        f.create_dataset("fit_time", data=fit_time)
        f.create_dataset("test_transform_time", data=test_transform_time)
        f.attrs['mode'] = mode
        
    del red_test

    # --- 5. Transform and Save Full Training Data ---
    
    start_time = time.time()
    
    if mode == 'normal':
        # Scenario 1: NORMAL Mode (Full data fits in RAM)
        train_full = np.array([])
        with h5py.File(dataset_path, "r") as f:
            train_full = f["train"][:]
            
        red_train = srp_object.transform(train_full)
        del train_full
        
        with h5py.File(save_path, "a") as f:
            f.create_dataset("red_train", data=red_train, dtype='f4')
            
        del red_train
        
    elif mode == 'constrained':
        # Scenario 2: CONSTRAINED Mode (Batch loading/transform)
        BATCH_SIZE = 10000 
        
        with h5py.File(save_path, "a") as f_out:
            dset_red_train = f_out.create_dataset("red_train", 
                                                  (num_total_samples, dim), 
                                                  dtype='f4')
            
            with h5py.File(dataset_path, "r") as f_in:
                dset_train_in = f_in["train"]
                
                for i in range(0, num_total_samples, BATCH_SIZE):
                    end_idx = min(i + BATCH_SIZE, num_total_samples)
                    
                    batch = dset_train_in[i:end_idx, :]
                    red_batch = srp_object.transform(batch)
                    dset_red_train[i:end_idx, :] = red_batch
                    
                    del batch
                    del red_batch
                    
    else:
        return

    end_time = time.time()
    transform_time = end_time - start_time
    
    with h5py.File(save_path, "a") as f:
        f.create_dataset("transform_time", data=transform_time)
        

def main():
    parser = argparse.ArgumentParser(description="Run dimensionality reduction on a dataset.")
    
    # Positional Arguments (required)
    parser.add_argument("dataset_path", type=str, 
        help="1. Path to the original HDF5 dataset file.")
    parser.add_argument("save_folder", type=str, 
        help="2. Path to the folder where the reduced HDF5 file will be saved.")
    parser.add_argument("dimension", type=int, 
        help="3. Target dimension for reduction.")
    parser.add_argument("sample_size", type=float, 
        help="4. Sample size (0.0 to 1.0) for fitting SRP.")
    
    # Optional Positional Argument with Default Value
    parser.add_argument("mode", type=str, 
        nargs='?', 
        choices=['normal', 'constrained'], 
        default='normal', 
        help="5. Processing mode: 'normal' or 'constrained'. Defaults to 'normal'.")
    
    args = parser.parse_args()
        
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    reduce_and_save(
        dataset_path=args.dataset_path,
        save_folder=args.save_folder,
        dim=args.dimension,
        sample_size=args.sample_size,
        mode=args.mode
    )


if __name__ == "__main__":
    main()