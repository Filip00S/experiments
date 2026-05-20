#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb:scratch_local=2gb
#PBS -l walltime=1:00:00

STORAGE_SERVER='YOUR_SERVER'
USERNAME='username'

ENV_NAME='experiments_env'
REPO_NAME='experiments/scripts'

REPO_DIR="/storage/${STORAGE_SERVER}/home/${USERNAME}/${REPO_NAME}"
LOGS_DIR="/storage/${STORAGE_SERVER}/home/${USERNAME}/${REPO_NAME}/logs"

module add conda-modules

cd "${REPO_DIR}" || {
    echo >&2 "Repository directory ${REPO_DIR} does not exist!"
    exit 1
}

conda activate "/storage/${STORAGE_SERVER}/home/${USERNAME}/.conda/envs/${ENV_NAME}" || {
    echo >&2 "Conda environment does not exist!"
    exit 2
}

# ARG format: "<experiments_dir> [--max_points <n>]"
python3 visualize_all.py $ARG
