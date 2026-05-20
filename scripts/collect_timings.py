import argparse
import os
import h5py
import pandas as pd
from collections import defaultdict


def build_df(data_dict):
    rows = []
    for outer_key, dim_dict in data_dict.items():
        row = dict(dim_dict)
        row["technique"] = outer_key
        rows.append(row)
    df = pd.DataFrame(rows)
    dim_cols = sorted([c for c in df.columns if c != "technique"], key=lambda x: int(x))
    return df[["technique"] + dim_cols]


def main():
    parser = argparse.ArgumentParser(description="Collect timing data from reduced HDF5 files.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset experiment directory.")
    parser.add_argument("folder", type=str, help="Subfolder to process (1, 2, or 3).")
    parser.add_argument("dataset_base_name", type=str, help="Base name of the dataset (without .hdf5).")
    args = parser.parse_args()

    target_dir = os.path.join(args.dataset_dir, args.folder)
    prefix = f"{args.dataset_base_name}_"

    fit_dict = defaultdict(dict)
    transform_dict = defaultdict(dict)
    test_transform_dict = defaultdict(dict)

    for fname in os.listdir(target_dir):
        if not fname.endswith(".hdf5") or not fname.startswith(prefix):
            continue

        experiment_part = fname[len(prefix):-5]
        parts = experiment_part.split("_")

        if len(parts) < 3:
            continue

        technique = parts[0]
        dim = parts[1]
        sample_size = parts[2]
        param = parts[3] if len(parts) == 4 else None

        outer_key = f"{technique}_{sample_size}_{param}" if param else f"{technique}_{sample_size}"

        fpath = os.path.join(target_dir, fname)
        with h5py.File(fpath, "r") as f:
            fit_dict[outer_key][dim] = float(f["fit_time"][()])
            transform_dict[outer_key][dim] = float(f["transform_time"][()])
            test_transform_dict[outer_key][dim] = float(f["test_transform_time"][()])

    tables_dir = os.path.join(target_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    base = f"{args.dataset_base_name}_{args.folder}"

    for timing_name, data_dict in [
        ("fit_time", fit_dict),
        ("transform_time", transform_dict),
        ("test_transform_time", test_transform_dict),
    ]:
        df = build_df(data_dict)
        out_path = os.path.join(tables_dir, f"{base}_{timing_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
