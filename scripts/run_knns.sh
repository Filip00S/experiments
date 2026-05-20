#!/bin/bash

EXPER="/storage/YOUR_SERVER/home/username/experiments"

DATASET_FILES=(
    "fashion-mnist-784-euclidean.hdf5"
    "laion2B-en-clip768v2-n=300K-cosine.hdf5"
    "sift-128-euclidean.hdf5"
    "gist-960-euclidean.hdf5"
)

DATASET_NAMES=(
    "fashion-mnist-784-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
    "sift-128-euclidean"
    "gist-960-euclidean"
)

NUM_DATASETS=${#DATASET_NAMES[@]}
for (( i=0; i<NUM_DATASETS; i++ )); do

    DATASET_FILE="${DATASET_FILES[$i]}"

    DATASET_PATH="${EXPER}/datasets/${DATASET_FILE}"

    export ARG="${DATASET_PATH}"
    qsub -v ARG compute_knns_original_gpu.sh

done
