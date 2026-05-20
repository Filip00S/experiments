#!/bin/bash

DATASET_PATH="$1"
PATH_1="$2"
PATH_2="$3"
DIM_LIST="$4"

for DIM in $DIM_LIST; do
    ARG_STRING_1="${DATASET_PATH} ${PATH_1} ${DIM} 1 normal"
    ARG_STRING_2="${DATASET_PATH} ${PATH_2} ${DIM} 0.4 normal"
    
    qsub -v ARG="${ARG_STRING_1}" pca12.sh
    qsub -v ARG="${ARG_STRING_1}" srp12.sh
    qsub -v ARG="${ARG_STRING_1}" grp12.sh
    qsub -v ARG="${ARG_STRING_1} 15" umap12.sh
    qsub -v ARG="${ARG_STRING_1} 40" umap12.sh
    if [ "$DIM" -lt 4 ]; then
        qsub -v ARG="${ARG_STRING_1} 15" tsne12.sh
        qsub -v ARG="${ARG_STRING_1} 40" tsne12.sh
    fi
    qsub -v ARG="${ARG_STRING_1} 50" ae12.sh
    qsub -v ARG="${ARG_STRING_1} 80" ae12.sh

    qsub -v ARG="${ARG_STRING_2}" pca12.sh
    qsub -v ARG="${ARG_STRING_2}" srp12.sh
    qsub -v ARG="${ARG_STRING_2}" grp12.sh
    qsub -v ARG="${ARG_STRING_2} 15" umap12.sh
    qsub -v ARG="${ARG_STRING_2} 40" umap12.sh
    if [ "$DIM" -lt 4 ]; then
        qsub -v ARG="${ARG_STRING_2} 15" tsne12.sh
        qsub -v ARG="${ARG_STRING_2} 40" tsne12.sh
    fi
    qsub -v ARG="${ARG_STRING_2} 50" ae12.sh
    qsub -v ARG="${ARG_STRING_2} 80" ae12.sh
done
