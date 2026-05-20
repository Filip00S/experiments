import os
import sys
import h5py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Find incomplete reduced HDF5 files.")
    parser.add_argument("--exper_dir", type=str, default="/storage/brno12-cerit/home/balko/experiments",
                        help="Base experiments directory")
    args = parser.parse_args()

    dataset_names = [
        "fashion-mnist-784-euclidean",
        "laion2B-en-clip768v2-n=300K-cosine",
        "sift-128-euclidean",
        "swiss_roll-euclidean",
        "s_curve-euclidean",
        "clusters-euclidean",
        "gist-960-euclidean"
    ]
    
    required_keys = ["red_train", "red_test", "red_train_neighbors", "red_neighbors"]
    incomplete_count = 0

    for ds_name in dataset_names:
        for folder in ["1", "2", "3"]:
            target_dir = os.path.join(args.exper_dir, ds_name, folder)
            
            if not os.path.exists(target_dir):
                continue
                
            for filename in os.listdir(target_dir):
                if filename.endswith(".hdf5"):
                    filepath = os.path.join(target_dir, filename)
                    is_complete = True
                    
                    try:
                        with h5py.File(filepath, "r") as f:
                            for key in required_keys:
                                if key not in f:
                                    is_complete = False
                                    break
                    except Exception as e:
                    # If the file is corrupted, unreadable, or open in another process
                        print(f"Could not open file {filepath}: {e}")
                        is_complete = False
                        
                    if not is_complete:
                        incomplete_count += 1
                        # Print only the full path to standard output
                        print(filepath)
                        
    print(f"\n--- Scan Complete: Found {incomplete_count} incomplete files ---", file=sys.stderr)

if __name__ == "__main__":
    main()