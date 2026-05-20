#!/bin/bash

MISSING_FILES="missing_files.txt"

if [ ! -f "$MISSING_FILES" ]; then
    echo "Error: $MISSING_FILES not found in the current directory!"
    exit 1
fi

echo "Submitting jobs for incomplete HDF5 files..."

while IFS= read -r REDUCED_FILE; do
    # Skip empty lines
    [ -z "$REDUCED_FILE" ] && continue
    
    if [ -f "$REDUCED_FILE" ]; then
        export ARG="${REDUCED_FILE//,/__COMMA__}"
        qsub -v ARG compute_knns_gpu.sh
    fi
done < "$MISSING_FILES"

echo "All missing files successfully submitted to the queue!"
