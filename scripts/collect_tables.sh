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

METRICS=(
    "np"
    "trustworthiness"
    "continuity"
    "mrre"
)

K_VALUES=(10 50 100)

echo "Starting table collection submission..."

for DS in "${DATASETS[@]}"; do
    DS_DIR="${EXPERIMENTS_DIR}/${DS}"

    for FOLDER in "1" "2" "3"; do
        TARGET_DIR="${DS_DIR}/${FOLDER}"

        if [ ! -d "$TARGET_DIR" ]; then
            continue
        fi

        for METRIC in "${METRICS[@]}"; do
            for K in "${K_VALUES[@]}"; do
                export ARG="${DS_DIR} ${FOLDER} ${METRIC} ${K} ${DS}"
                qsub -v ARG collect_table_job.sh
            done
        done
    done
done

echo "All table collection jobs submitted!"
