#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:30:00
#PBS -o /storage/YOUR_SERVER/home/username/experiments/scripts/logs/
#PBS -e /storage/YOUR_SERVER/home/username/experiments/scripts/logs/

STORAGE_SERVER='YOUR_SERVER'
USERNAME='username'

ENV_NAME='experiments_env'
REPO_NAME='experiments/scripts'

REPO_DIR="/storage/${STORAGE_SERVER}/home/${USERNAME}/${REPO_NAME}"

module add conda-modules

cd "${REPO_DIR}" || {
    echo >&2 "Repository directory ${REPO_DIR} does not exist!"
    exit 1
}

conda activate "/storage/${STORAGE_SERVER}/home/${USERNAME}/.conda/envs/${ENV_NAME}" || {
    echo >&2 "Conda environment does not exist!"
    exit 2
}

echo "Scanning for incomplete HDF5 files..."
python3 find_incomplete_hdf5.py > missing_files.txt
echo "Done. The list of incomplete files has been saved to missing_files.txt in your scripts directory."