#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb:cl_konos=False
#PBS -l walltime=00:05:00

STORAGE_SERVER='YOUR_SERVER'
USERNAME='username'
ENV_NAME='experiments_env'
REPO_DIR="/storage/${STORAGE_SERVER}/home/${USERNAME}/experiments/scripts"

module add conda-modules

cd "${REPO_DIR}" || {
    echo >&2 "Repository directory ${REPO_DIR} does not exist!"
    exit 1
}

conda activate "/storage/${STORAGE_SERVER}/home/${USERNAME}/.conda/envs/${ENV_NAME}" || {
    echo >&2 "Conda environment does not exist!"
    exit 2
}

if [ -z "$ARG" ]; then
    echo "Missing ARG environment variable!"
    exit 3
fi

read -r DATASET_DIR FOLDER DATASET_BASE_NAME <<< "$ARG"

python3 -u collect_timings.py "${DATASET_DIR}" "${FOLDER}" "${DATASET_BASE_NAME}"
