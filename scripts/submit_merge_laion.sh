#!/bin/bash

TARGET_DIR="/storage/YOUR_SERVER/home/username/experiments/datasets"

OLD_FILE="$TARGET_DIR/laion2B-en-clip768v2-n=300K.h5"
LAION_TRAIN="$TARGET_DIR/laion2B-en-clip768v2-n300K.h5"
LAION_TEST="$TARGET_DIR/public-queries-10k-clip768v2.h5"
LAION_OUT="$TARGET_DIR/laion2B-en-clip768v2-n=300K.hdf5"

if [ -f "$OLD_FILE" ]; then
    echo "Renaming ugly dataset file to clean format..."
    mv "$OLD_FILE" "$LAION_TRAIN"
fi

qsub -v TRAIN_FILE="$LAION_TRAIN",TEST_FILE="$LAION_TEST",OUT_FILE="$LAION_OUT" merge_laion_job.sh

echo "Submitted LAION merge job to PBS."
