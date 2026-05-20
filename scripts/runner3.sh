#!/bin/bash

DATASET_PATH="$1"
PATH_1="$2"
DIM_LIST="$3"

# 3. Option: Training in restricted (constrained) mode on a sample
# Memory limit is 4GB, so sample sizes are carefully adjusted per algorithm.

for DIM in $DIM_LIST; do
    # Light techniques: sample sizes 0.1 (10%) and 0.2 (20%) for PCA, 0.2 (20%) and 0.4 (40%) for SRP/GRP
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.1 constrained" pca3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.2 constrained" pca3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.2 constrained" srp3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.4 constrained" srp3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.2 constrained" grp3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.4 constrained" grp3.sh

    # High-memory techniques (Graph/Manifold based): sample sizes 0.01 (1%) and 0.05 (5%)
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.01 constrained 15" umap3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.05 constrained 15" umap3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.01 constrained 40" umap3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.05 constrained 40" umap3.sh

    if [ "$DIM" -lt 4 ]; then
        qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.01 constrained 15" tsne3.sh
        qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.05 constrained 15" tsne3.sh
        qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.01 constrained 40" tsne3.sh
        qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.05 constrained 40" tsne3.sh
    fi

    # Deep learning (PyTorch) technique: sample sizes 0.2 (20%) and 0.4 (40%)
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.2 constrained 50" ae3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.2 constrained 80" ae3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.4 constrained 50" ae3.sh
    qsub -v ARG="${DATASET_PATH} ${PATH_1} ${DIM} 0.4 constrained 80" ae3.sh
done