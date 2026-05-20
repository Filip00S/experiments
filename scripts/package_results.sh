#!/bin/bash

EXPERIMENTS_DIR="/storage/YOUR_SERVER/home/username/experiments"
ARCHIVE="$(pwd)/results_export.tar.gz"
COLLECT_DIRS=("visualize" "tables")

DATASETS=(
    "fashion-mnist-784-euclidean"
    "sift-128-euclidean"
    "swiss_roll-euclidean"
    "s_curve-euclidean"
    "clusters-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
    "gist-960-euclidean"
)

PATHS=()

for DS in "${DATASETS[@]}"; do
    for FOLDER in "1" "2" "3"; do
        for COLLECT in "${COLLECT_DIRS[@]}"; do
            REL="${DS}/${FOLDER}/${COLLECT}"
            if [ -d "${EXPERIMENTS_DIR}/${REL}" ]; then
                PATHS+=("$REL")
                echo "  Found: ${REL}"
            fi
        done
    done
done

# Original dataset visualizations saved by visualize_all.py into datasets/visualize/
DATASETS_VIZ="${EXPERIMENTS_DIR}/datasets/visualize"
if [ -d "$DATASETS_VIZ" ]; then
    PATHS+=("datasets/visualize")
    echo "  Found: datasets/visualize"
fi

if [ ${#PATHS[@]} -eq 0 ]; then
    echo "No matching directories found. Nothing to archive."
    exit 1
fi

echo "Creating archive: ${ARCHIVE}"
tar -czf "$ARCHIVE" -C "$EXPERIMENTS_DIR" "${PATHS[@]}"

echo "Done: ${ARCHIVE}"
