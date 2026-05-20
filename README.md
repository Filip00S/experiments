# 📐 Nearest-Neighbor Ordering Under Dimensionality Reduction — Experiments

> Reproducibility repository for the bachelor's thesis *Nearest-Neighbor Ordering Under Dimensionality Reduction* (Filip Balko, FI MU, 2026).

The pipeline evaluates **6 dimensionality reduction techniques** across **7 datasets** using **4 neighborhood-preservation metrics**.

| 🔧 Techniques | 📊 Metrics |
|---|---|
| PCA, Gaussian RP, Sparse RP, UMAP, t-SNE, SNN Autoencoder | Neighborhood Precision, Trustworthiness, Continuity, MRRE |

All experiments were run on [MetaCentrum](https://metavo.metacentrum.cz) (Czech national grid) via the PBS job scheduler. Every `qsub` command in the scripts assumes that environment.

---

## 📁 Repository layout

```
experiments/
├── scripts/
│   ├── create_*.py              # Synthetic dataset generators
│   ├── dataset_neighbors*.py    # k-NN computation (CPU & GPU)
│   ├── experiment_*.py          # Reduction technique implementations
│   ├── evaluate_*.py            # Quality metric scripts
│   ├── collect_*.py             # Result aggregation
│   ├── visualize_all.py
│   ├── find_incomplete_hdf5.py
│   ├── logs/                    # PBS stdout/stderr
│   └── *.sh                     # PBS job wrappers and submission drivers
├── datasets/                    # Raw HDF5 files (downloaded in Step 1)
├── <dataset-name>/
│   ├── 1/                       # Mode 1 — full training set (sample = 1.0)
│   ├── 2/                       # Mode 2 — sampled training set (sample = 0.4)
│   └── 3/                       # Mode 3 — constrained mode (4 GB RAM)
└── requirements.txt
```

---

## ⚙️ Prerequisites

- A MetaCentrum node with PBS (`qsub`), or equivalent HPC cluster
- Conda accessible via `module add conda-modules`
- Python 3.10+

### Create the Conda environment

```bash
conda create -n experiments_env python=3.10
conda activate experiments_env
pip install -r requirements.txt
```

> `requirements.txt` pins `torch==2.11.0` built against CUDA 13 (`cu130`). The SNN Autoencoder
> and GPU k-NN scripts require CUDA. If your cluster uses a different CUDA version, replace
> the `--extra-index-url` suffix and the torch version accordingly.
> See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## 🔩 Configuration

Every shell script hard-codes the storage server and username. Update them before running:

| Variable | Default | Description |
|---|---|---|
| `STORAGE_SERVER` | `YOUR_SERVER` | MetaCentrum storage server |
| `USERNAME` | `username` | Your MetaCentrum username |
| `EXPER` | `/storage/YOUR_SERVER/home/username/experiments` | Absolute path to this repo on storage |

Apply globally with a single command:

```bash
cd scripts/
sed -i 's|YOUR_SERVER|YOUR_SERVER|g; s|username|YOUR_USERNAME|g' *.sh
```

---

## 🗺️ Pipeline overview

```
Step 0   🔐  Permissions & directories
Step 1   📥  Dataset acquisition
Step 2   🔀  Preprocessing
Step 3   🔍  Ground-truth k-NN  (original space)
Step 4   📉  Dimensionality reduction  ← main experiment step
Step 5   🔍  Reduced-space k-NN
Step 6   🩺  Completeness check
Step 7   📊  Evaluation
Step 8   📋  Result collection
Step 9   🎨  Visualization
```

---

## 📖 Step-by-step guide

### Step 0 — 🔐 Permissions and directories

Run once after cloning. Makes every `.sh` executable and creates `logs/` and `datasets/`.

```bash
cd scripts/
bash grant_permissions.sh
```

---

### Step 1 — 📥 Dataset acquisition

#### 1a. Real-world benchmark datasets

```bash
bash download_datasets.sh
```

Downloads five files into `datasets/` (`curl -L -C -` resumes interrupted downloads):

| File | Source | Size |
|---|---|---|
| `fashion-mnist-784-euclidean.hdf5` | ann-benchmarks.com | ~480 MB |
| `sift-128-euclidean.hdf5` | ann-benchmarks.com | ~510 MB |
| `gist-960-euclidean.hdf5` | ann-benchmarks.com | ~3.6 GB |
| `laion2B-en-clip768v2-n=300K.h5` | HuggingFace / SISAP 2023 | ~900 MB |
| `public-queries-10k-clip768v2.h5` | HuggingFace / SISAP 2023 | ~30 MB |

#### 1b. Synthetic datasets

```bash
bash run_create_synthetic.sh
```

Submits 3 PBS jobs that generate 3-D manifold datasets (10 000 train / 1 000 test points each):

- `datasets/swiss_roll-euclidean.hdf5`
- `datasets/s_curve-euclidean.hdf5`
- `datasets/clusters-euclidean.hdf5`

---

### Step 2 — 🔀 Preprocessing

The LAION download is split into two files. Merge them:

```bash
bash submit_merge_laion.sh
```

Produces `datasets/laion2B-en-clip768v2-n=300K-cosine.hdf5`. All other datasets need no preprocessing — the `-euclidean` / `-cosine` suffix in each filename encodes the distance metric used throughout the pipeline.

---

### Step 3 — 🔍 Ground-truth k-NN (original space)

Compute exact 101-nearest neighbors on every original dataset using GPU-accelerated PyTorch search. Results are appended to the same HDF5 file as `neighbors` (test→train) and `train_neighbors` (train→train).

```bash
bash run_knns.sh
```

Submits one `compute_knns_original_gpu.sh` job per dataset (**1 GPU, 2 CPUs, 8 GB RAM, 1 h**).

---

### Step 4 — 📉 Dimensionality reduction experiments

```bash
bash run_all_exp.sh
```

Calls `runner12.sh` for the six non-GIST datasets (Modes 1 & 2) and `runner3.sh` for GIST (Mode 3). Both runners iterate over all target dimensions and submit one PBS job per technique per dimension.

#### Execution modes

| Mode | Runner | `sample_size` | `mode` arg | RAM | Description |
|---|---|---|---|---|---|
| **1** | `runner12.sh` | `1.0` | `normal` | 16 GB | Full training set |
| **2** | `runner12.sh` | `0.4` | `normal` | 16 GB | Random 40% subsample |
| **3** | `runner3.sh` | 0.01 – 0.4 | `constrained` | 4 GB | Batched transform, per-technique sample fraction |

#### Target dimensions per dataset

| Dataset | Target dimensions |
|---|---|
| Fashion-MNIST (784-d) | 3, 90, 190, 290, 410, 530, 680 |
| LAION-2B 300K (768-d) | 16, 32, 64, 96, 128, 256, 512 |
| SIFT1M (128-d) | 2, 4, 8, 16, 32, 64, 120 |
| GIST1M (960-d) | 2, 16, 32, 64, 128, 256, 512, 768 |
| Swiss Roll / S-Curve / Clusters (3-d) | 2 |

#### Techniques

| Script | Technique | Hyperparameter sweep |
|---|---|---|
| `experiment_pca.py` | PCA / SVD | — |
| `experiment_grp.py` | Gaussian Random Projection | — |
| `experiment_srp.py` | Sparse Random Projection | — |
| `experiment_umap.py` | UMAP | `n_neighbors` ∈ {15, 40} |
| `experiment_tsne.py` | t-SNE | `perplexity` ∈ {15, 40} (target dim < 4 only) |
| `experiment_ae.py` | SNN Autoencoder | `k` ∈ {50, 80} |

Each script signature:
```
python3 experiment_<X>.py <dataset.hdf5> <output_dir> <target_dim> <sample_size> <mode> [param]
```

Output is written as `<dataset-name>/1/<dataset>_PCA_<dim>_1,0.hdf5`, containing `red_train`, `red_test`, `fit_time`, and `transform_time`. An early-exit guard skips files that already exist.

---

### Step 5 — 🔍 Reduced-space k-NN

Recompute exact neighbors on the reduced vectors:

```bash
bash run_lower_knns.sh
```

Iterates over all reduced HDF5 files in mode folders `1` and `2` (non-GIST) and folder `3` (GIST), submitting one `compute_knns_gpu.sh` job per file (**1 GPU, 2 CPUs, 8 GB RAM, 1 h**). Appends `red_train_neighbors` and `red_neighbors` to each file in place.

**Retrying failed jobs:** pipe incomplete file paths into `run_missing_knns.sh`:

```bash
python3 scripts/find_incomplete_hdf5.py --exper_dir /path/to/experiments > missing_files.txt
bash run_missing_knns.sh
```

---

### Step 6 — 🩺 Completeness check

Scan for any reduced HDF5 file missing a required key (`red_train`, `red_test`, `red_train_neighbors`, `red_neighbors`):

```bash
python3 scripts/find_incomplete_hdf5.py --exper_dir /path/to/experiments
```

Prints incomplete paths to stdout, summary count to stderr.

---

### Step 7 — 📊 Evaluation

```bash
bash run_evaluations.sh
```

Submits one `eval.sh` job per `(dataset, folder, reduced file, metric, k)` tuple (**1 CPU, 4 GB RAM, 30 min**). Each job copies both HDF5 files to local NVMe scratch, runs the metric script, and copies the resulting CSV back.

**Metrics:** Neighborhood Precision, Trustworthiness, Continuity, MRRE  
**k values:** 10, 50, 100

Output CSV naming: `<dataset>_<technique>_<dim>_<sample>_<metric>_k=<K>.csv`

---

### Step 8 — 📋 Result collection

```bash
bash collect_tables.sh    # pivot metric CSVs into summary tables per (dataset, folder, metric, k)
bash collect_timings.sh   # collect fit_time / transform_time from HDF5 files
```

Tables are written to `<dataset>/<folder>/tables/` as separate train / test CSVs.

---

### Step 9 — 🎨 Visualization

```bash
bash run_visualize.sh
```

Generates PDF scatter plots for the three synthetic datasets (Swiss Roll, S-Curve, Clusters). Produces one PDF per reduced file (plus a query-overlay version) in `<dataset>/<folder>/visualize/`, and original 3-D plots in `datasets/visualize/`.

To run locally without PBS:

```bash
python3 scripts/visualize_all.py /path/to/experiments
```

---

## ⚡ Quick-reference execution order

| Step | Command | Notes |
|---|---|---|
| 0 🔐 | `bash grant_permissions.sh` | Run once |
| 1a 📥 | `bash download_datasets.sh` | |
| 1b 🧪 | `bash run_create_synthetic.sh` | Wait for PBS |
| 2 🔀 | `bash submit_merge_laion.sh` | Wait for PBS |
| 3 🔍 | `bash run_knns.sh` | Wait for PBS |
| 4 📉 | `bash run_all_exp.sh` | Wait for PBS — long step |
| 5 🔍 | `bash run_lower_knns.sh` | Wait for PBS |
| 6 🩺 | `python3 find_incomplete_hdf5.py ...` | Optional — retry with `run_missing_knns.sh` |
| 7 📊 | `bash run_evaluations.sh` | Wait for PBS |
| 8 📋 | `bash collect_tables.sh && bash collect_timings.sh` | |
| 9 🎨 | `bash run_visualize.sh` | |

---

## 🗂️ Output structure after a complete run

```
experiments/
├── datasets/
│   ├── fashion-mnist-784-euclidean.hdf5      # train, test, neighbors, train_neighbors
│   ├── sift-128-euclidean.hdf5
│   ├── gist-960-euclidean.hdf5
│   ├── laion2B-en-clip768v2-n=300K-cosine.hdf5
│   ├── swiss_roll-euclidean.hdf5
│   ├── s_curve-euclidean.hdf5
│   ├── clusters-euclidean.hdf5
│   └── visualize/
├── fashion-mnist-784-euclidean/
│   ├── 1/
│   │   ├── *_PCA_3_1,0.hdf5                  # red_train, red_test, fit_time,
│   │   │                                     # transform_time, red_*_neighbors
│   │   ├── *_np_k=10.csv
│   │   ├── tables/
│   │   │   ├── *_train_np_k=10.csv
│   │   │   └── ...
│   │   └── visualize/
│   └── 2/
├── gist-960-euclidean/
│   └── 3/
└── ...
```

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](LICENSE).
