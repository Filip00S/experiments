#!/bin/bash

EXPER="/storage/YOUR_SERVER/home/username/experiments"
OUTPUT_DIR="${EXPER}/datasets"

qsub -v PYTHON_SCRIPT="create_swiss_roll.py",OUTPUT_DIR="${OUTPUT_DIR}" create_dataset_job.sh
qsub -v PYTHON_SCRIPT="create_s_curve.py",OUTPUT_DIR="${OUTPUT_DIR}" create_dataset_job.sh
qsub -v PYTHON_SCRIPT="create_clusters.py",OUTPUT_DIR="${OUTPUT_DIR}" create_dataset_job.sh

echo "Submitted PBS jobs to create Swiss Roll, S-Curve, and Clusters datasets."