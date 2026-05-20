#!/bin/bash

# TODO: Initialize the target directory for downloads
HOME_DIR="/storage/YOUR_SERVER/home/username"

# Array of dataset URLs (to be populated)
DATASET_URLS=(
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
    "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    "https://huggingface.co/datasets/Coda-Research-Group/SISAP_2023_Indexing_Challenge/resolve/main/laion2B-en-clip768v2-n%3D300K.h5"
    "https://huggingface.co/datasets/Coda-Research-Group/SISAP_2023_Indexing_Challenge/resolve/main/public-queries-10k-clip768v2.h5"
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
)

# Safety check to ensure HOME_DIR is set
if [ -z "$HOME_DIR" ]; then
    echo "Error: HOME_DIR is not initialized. Please set it before running the script."
    exit 1
fi

TARGET_DIR="${HOME_DIR}/experiments/datasets"

# Create the target directory if it doesn't already exist
mkdir -p "$TARGET_DIR"

# Iterate over the URLs and download them
for URL in "${DATASET_URLS[@]}"; do
    echo "Downloading $URL into $TARGET_DIR..."
    FILENAME=$(basename "$URL")
    # Replace URL-encoded %3D with =
    FILENAME="${FILENAME//%3D/=}"
    curl -L -C - -o "$TARGET_DIR/$FILENAME" "$URL"
done

echo "All downloads finished."
