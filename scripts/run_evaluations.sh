#!/bin/bash

# Base directory configurations
EXPERIMENTS_DIR="/storage/YOUR_SERVER/home/username/experiments"
SCRIPTS_DIR="${EXPERIMENTS_DIR}/scripts"
DATASETS_DIR="${EXPERIMENTS_DIR}/datasets"

DATASETS=(
    "fashion-mnist-784-euclidean"
    "sift-128-euclidean"
    "swiss_roll-euclidean"
    "s_curve-euclidean"
    "clusters-euclidean"
    "laion2B-en-clip768v2-n=300K-cosine"
)

SCRIPTS=(
    "evaluate_np.py"
    "evaluate_trustworthiness.py"
    "evaluate_continuity.py"
    "evaluate_mrre.py"
)

K_VALUES=(10 50 100)

echo "Starting massive evaluation submission..."

for DS in "${DATASETS[@]}"; do
    ORIGINAL="${DATASETS_DIR}/${DS}.hdf5"
    
    if [ ! -f "$ORIGINAL" ]; then
        echo "Warning: Original dataset not found at $ORIGINAL. Skipping..."
        continue
    fi

    for FOLDER in "1" "2"; do
        TARGET_DIR="${EXPERIMENTS_DIR}/${DS}/${FOLDER}"
        
        if [ ! -d "$TARGET_DIR" ]; then
            continue
        fi
        
        # Iterate over all reduced HDF5 files
        for REDUCED in "${TARGET_DIR}"/*.hdf5; do
            [ -e "$REDUCED" ] || continue
            REDUCED_BASENAME=$(basename "$REDUCED" .hdf5)
            
            for SCRIPT in "${SCRIPTS[@]}"; do
                # Extract metric name (e.g. 'np' from 'evaluate_np.py')
                METRIC="${SCRIPT#evaluate_}"
                METRIC="${METRIC%.py}"
                
                for K in "${K_VALUES[@]}"; do
                    # Check if output CSV already exists to avoid redundant jobs
                    # if [ -f "${TARGET_DIR}/${REDUCED_BASENAME}_${METRIC}_k=${K}.csv" ]; then
                    #     continue
                    # fi

                    SAFE_REDUCED="${REDUCED//,/__COMMA__}"
                    export ARG="${SCRIPT} ${ORIGINAL} ${SAFE_REDUCED} ${K}"
                    qsub -v ARG eval.sh
                done
            done
        done
    done
done

# ---------------------------------------------------------
# GIST Dataset (Isolated Execution for directory /3)
# ---------------------------------------------------------
GIST_DS="gist-960-euclidean"
GIST_ORIGINAL="${DATASETS_DIR}/${GIST_DS}.hdf5"

if [ -f "$GIST_ORIGINAL" ]; then
    GIST_TARGET_DIR="${EXPERIMENTS_DIR}/${GIST_DS}/3"
    
    if [ -d "$GIST_TARGET_DIR" ]; then
        for REDUCED in "${GIST_TARGET_DIR}"/*.hdf5; do
            [ -e "$REDUCED" ] || continue
            REDUCED_BASENAME=$(basename "$REDUCED" .hdf5)

            for SCRIPT in "${SCRIPTS[@]}"; do
                METRIC="${SCRIPT#evaluate_}"
                METRIC="${METRIC%.py}"

                for K in "${K_VALUES[@]}"; do
                    # if [ -f "${GIST_TARGET_DIR}/${REDUCED_BASENAME}_${METRIC}_k=${K}.csv" ]; then
                    #     continue
                    # fi

                    SAFE_REDUCED="${REDUCED//,/__COMMA__}"
                    export ARG="${SCRIPT} ${GIST_ORIGINAL} ${SAFE_REDUCED} ${K}"
                    qsub -v ARG eval.sh
                done
            done
        done
    fi
else
    echo "Warning: Original dataset not found at $GIST_ORIGINAL. Skipping..."
fi

echo "All evaluation jobs successfully submitted to the PBS queue!"