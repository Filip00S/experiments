#!/bin/bash

EXPER="/storage/YOUR_SERVER/home/username/experiments"

qsub -v ARG="${EXPER}" visualize_job.sh

echo "Submitted visualization job for swiss_roll, s_curve, and clusters."
