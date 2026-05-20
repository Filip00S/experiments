#!/bin/bash
EXPER="/storage/YOUR_SERVER/home/username/experiments"

DATASET_NAMES=(
    "fashion-mnist-784-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
    "sift-128-euclidean"
    "swiss_roll-euclidean"
    "s_curve-euclidean"
    "clusters-euclidean"
)

NUM_DATASETS=${#DATASET_NAMES[@]}
for (( i=0; i<NUM_DATASETS; i++ )); do

    DATASET_NAME="${DATASET_NAMES[$i]}"

    for MODE_FOLDER in "1" "2"; do
        TARGET_DIR="${EXPER}/${DATASET_NAME}/${MODE_FOLDER}"
        for REDUCED_FILE in "${TARGET_DIR}"/*.hdf5; do
            if [ -f "$REDUCED_FILE" ]; then
                export ARG="${REDUCED_FILE//,/__COMMA__}"
                qsub -v ARG compute_knns_gpu.sh
            fi
        done
    done
done

# ---------------------------------------------------------
# GIST Dataset (Isolated Execution for directory /3)
# ---------------------------------------------------------
GIST_DS="gist-960-euclidean"
GIST_TARGET_DIR="${EXPER}/${GIST_DS}/3"

if [ -d "$GIST_TARGET_DIR" ]; then
    for REDUCED_FILE in "${GIST_TARGET_DIR}"/*.hdf5; do
        if [ -f "$REDUCED_FILE" ]; then
            export ARG="${REDUCED_FILE//,/__COMMA__}"
            qsub -v ARG compute_knns_gpu.sh
        fi
    done
fi
