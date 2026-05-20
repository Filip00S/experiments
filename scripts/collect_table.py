import argparse
import os
import csv
import pandas as pd
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation CSVs into summary tables.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset experiment directory.")
    parser.add_argument("folder", type=str, help="Subfolder to process (1, 2, or 3).")
    parser.add_argument("metric", type=str, help="Metric name (np, trustworthiness, continuity, mrre).")
    parser.add_argument("k", type=int, help="K value.")
    parser.add_argument("dataset_base_name", type=str, help="Base name of the dataset (without .hdf5).")
    args = parser.parse_args()

    target_dir = os.path.join(args.dataset_dir, args.folder)
    prefix = f"{args.dataset_base_name}_"
    suffix = f"_{args.metric}_k={args.k}.csv"

    train_dict = defaultdict(dict)
    test_dict = defaultdict(dict)

    for fname in os.listdir(target_dir):
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue

        experiment_part = fname[len(prefix):-len(suffix)]
        parts = experiment_part.split("_")

        if len(parts) < 3:
            continue

        technique = parts[0]
        dim = parts[1]
        sample_size = parts[2]
        param = parts[3] if len(parts) == 4 else None

        outer_key = f"{technique}_{sample_size}_{param}" if param else f"{technique}_{sample_size}"

        fpath = os.path.join(target_dir, fname)
        with open(fpath, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            row = next(reader)
            train_val = float(row["train"])
            test_val = float(row["test"])

        train_dict[outer_key][dim] = train_val
        test_dict[outer_key][dim] = test_val

    def build_df(data_dict):
        rows = []
        for outer_key, dim_dict in data_dict.items():
            row = dict(dim_dict)
            row["technique"] = outer_key
            rows.append(row)
        df = pd.DataFrame(rows)
        dim_cols = sorted([c for c in df.columns if c != "technique"], key=lambda x: int(x))
        return df[["technique"] + dim_cols]

    train_df = build_df(train_dict)
    test_df = build_df(test_dict)

    tables_dir = os.path.join(target_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    base = f"{args.dataset_base_name}_{args.folder}"
    train_path = os.path.join(tables_dir, f"{base}_train_{args.metric}_k={args.k}.csv")
    test_path = os.path.join(tables_dir, f"{base}_test_{args.metric}_k={args.k}.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
