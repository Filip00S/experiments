#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=10gb:cl_konos=False
#PBS -l walltime=00:30:00

STORAGE_SERVER='YOUR_SERVER'
USERNAME='username'
ENV_NAME='experiments_env'
REPO_DIR="/storage/${STORAGE_SERVER}/home/${USERNAME}/experiments/scripts"
LOGS_DIR="${REPO_DIR}/logs"

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

# Parse parameters passed via ARG
read -r SCRIPT ORIGINAL REDUCED K <<< "$ARG"

# Restore any commas that were stripped for PBS qsub parsing compatibility
REDUCED="${REDUCED//__COMMA__/,}"

# --- NEW: Caching to Local NVMe Scratch ---
if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR" ]; then
    echo "Caching HDF5 datasets to local NVMe $SCRATCHDIR..."
    cp "${ORIGINAL}" "${SCRATCHDIR}/" || { echo "Failed to copy ORIGINAL"; exit 4; }
    cp "${REDUCED}" "${SCRATCHDIR}/" || { echo "Failed to copy REDUCED"; exit 5; }
    
    LOCAL_ORIGINAL="${SCRATCHDIR}/$(basename "${ORIGINAL}")"
    LOCAL_REDUCED="${SCRATCHDIR}/$(basename "${REDUCED}")"
else
    echo "Warning: SCRATCHDIR not found, falling back to NFS..."
    LOCAL_ORIGINAL="${ORIGINAL}"
    LOCAL_REDUCED="${REDUCED}"
fi

# Disable HDF5 file locking which causes major issues on distributed filesystems
export HDF5_USE_FILE_LOCKING=FALSE

REDUCED_BASENAME=$(basename "${REDUCED}" .hdf5)

echo "Running ${SCRIPT} with k=${K} on ${REDUCED_BASENAME}..."
python3 -u "${SCRIPT}" "${LOCAL_ORIGINAL}" "${LOCAL_REDUCED}" --k "${K}"

# --- NEW: Copy Results Back ---
if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR" ]; then
    echo "Moving results from scratch to storage..."
    REDUCED_DIR=$(dirname "${REDUCED}")
    cp "${SCRATCHDIR}"/*.csv "${REDUCED_DIR}/" || { echo "Failed to copy CSVs back to storage!"; exit 6; }
fi