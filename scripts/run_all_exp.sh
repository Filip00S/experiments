#!/bin/bash

EXPER="/storage/YOUR_SERVER/home/username/experiments"

DATASET_FILES=(
    "fashion-mnist-784-euclidean.hdf5"
    "laion2B-en-clip768v2-n=300K-cosine.hdf5"
    "sift-128-euclidean.hdf5"
    "swiss_roll-euclidean.hdf5"
    "s_curve-euclidean.hdf5"
    "clusters-euclidean.hdf5"
)

DATASET_NAMES=(
    "fashion-mnist-784-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
    "sift-128-euclidean"
    "swiss_roll-euclidean"
    "s_curve-euclidean"
    "clusters-euclidean"
)

DIM_LISTS=(
    "3 90 190 290 410 530 680"
    "16 32 64 96 128 256 512"
    "2 4 8 16 32 64 120"
    "2"
    "2"
    "2"
)

NUM_DATASETS=${#DATASET_NAMES[@]}
for (( i=0; i<NUM_DATASETS; i++ )); do
    
    DATASET_FILE="${DATASET_FILES[$i]}"
    DATASET_NAME="${DATASET_NAMES[$i]}"
    DIM_LIST="${DIM_LISTS[$i]}"
    
    DATASET_PATH="${EXPER}/datasets/${DATASET_FILE}"
    
    ARG2_PATH="${EXPER}/${DATASET_NAME}/1"
    ARG3_PATH="${EXPER}/${DATASET_NAME}/2"
    
    bash ./runner12.sh "$DATASET_PATH" "$ARG2_PATH" "$ARG3_PATH" "$DIM_LIST"
    
done

# ---------------------------------------------------------
# GIST Dataset (Isolated Execution)
# ---------------------------------------------------------
GIST_FILES=(
    "gist-960-euclidean.hdf5"
)

GIST_NAMES=(
    "gist-960-euclidean"
)

GIST_DIMS=(
    "2 16 32 64 128 256 512 768"
)

NUM_GIST=${#GIST_NAMES[@]}
for (( i=0; i<NUM_GIST; i++ )); do
    DATASET_PATH="${EXPER}/datasets/${GIST_FILES[$i]}"
    ARG3_PATH="${EXPER}/${GIST_NAMES[$i]}/3"
    
    bash ./runner3.sh "$DATASET_PATH" "$ARG3_PATH" "${GIST_DIMS[$i]}"
done