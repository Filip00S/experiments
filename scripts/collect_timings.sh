#!/bin/bash

EXPERIMENTS_DIR="/storage/YOUR_SERVER/home/username/experiments"

DATASETS=(
    "fashion-mnist-784-euclidean"
    "sift-128-euclidean"
    "swiss_roll-euclidean"
    "s_curve-euclidean"
    "clusters-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
    "gist-960-euclidean"
)

echo "Starting timing collection submission..."

for DS in "${DATASETS[@]}"; do
    DS_DIR="${EXPERIMENTS_DIR}/${DS}"

    for FOLDER in "1" "2" "3"; do
        TARGET_DIR="${DS_DIR}/${FOLDER}"

        if [ ! -d "$TARGET_DIR" ]; then
            continue
        fi

        export ARG="${DS_DIR} ${FOLDER} ${DS}"
        qsub -v ARG collect_timings_job.sh
    done
done

echo "All timing collection jobs submitted!"
