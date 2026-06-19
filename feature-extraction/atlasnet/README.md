# Feature Extraction

Scripts for extracting DNN features from 3D point clouds using AtlasNet autoencoder.

## Setup

### Install packages

```bash
sudo apt install cmake \
    libgmp-dev libmpfr-dev \
    libboost-dev libboost-filesystem-dev libboost-thread-dev \
    libeigen3-dev libcgal-dev
```

### Python environment

This directory uses [uv](https://docs.astral.sh/uv/) to manage the Python environment.
`torch` and `torchvision` are installed from prebuilt wheels; the correct wheel URL depends on your CUDA version.

**1. Check your CUDA version**

```bash
nvidia-smi | grep "CUDA Version"
# or
nvcc --version
```

**2. Edit `pyproject.toml` — set the wheel URLs to match your CUDA version**

The `[tool.uv.sources]` section specifies direct wheel URLs.
Replace the `cu111` tag with the tag for your CUDA version:

| CUDA | Tag | torch wheel | torchvision wheel |
|------|-----|-------------|-------------------|
| 10.2 | `cu102` | `torch-1.9.1%2Bcu102-cp38-cp38-linux_x86_64.whl` | `torchvision-0.10.1%2Bcu102-cp38-cp38-linux_x86_64.whl` |
| 11.1 | `cu111` | `torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl` | `torchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl` |
| 11.3 | `cu113` | `torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl` | `torchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl` |
| CPU only | `cpu` | `torch-1.9.1%2Bcpu-cp38-cp38-linux_x86_64.whl` | `torchvision-0.10.1%2Bcpu-cp38-cp38-linux_x86_64.whl` |

Base URL: `https://download.pytorch.org/whl/{tag}/`

Example for CUDA 11.3:

```toml
[tool.uv.sources]
torch = { url = "https://download.pytorch.org/whl/cu113/torch-1.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl" }
torchvision = { url = "https://download.pytorch.org/whl/cu113/torchvision-0.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl" }
```

> **Note:** torch 1.9.1 does not have a dedicated CUDA 11.3 build;
> use the `cu111` wheel (it is compatible with CUDA 11.x).

**3. Install**

```bash
uv sync
```

---

Point cloud files (`.npy`) must be placed under `data/pointcloud/` before running.

```
data/pointcloud/
├── train-3d-natural-objects/   *.npy
├── test-3d-natural-objects/    *.npy
└── test-3d-artificial-objects/  *.npy
```

Model weights and options must be placed at:

```
data/models/atlasnet/
├── network_crtd.pth
└── options.json
```

Scripts can be run from any directory (paths are resolved from `__file__`).

## Usage

### Step 1: Extract features

```bash
python extract_features_atlasnet.py
```

Extracts encoder features from point clouds for all datasets.
Already-existing output directories are skipped.

Output: `data/features/{dataset}/atlasnet/{layer}/{label}.mat`

| Dataset | Output directory |
|---|---|
| Train 3D Natural Objects | `data/features/train-3d-natural-objects/atlasnet/` |
| Test 3D Natural Objects | `data/features/test-3d-natural-objects/atlasnet/` |
| Test 3D Artificial Objects | `data/features/test-3d-artificial-objects/atlasnet/` |

Extracted layers: `encoder_conv1`, `encoder_conv2`, `encoder_conv3`, `encoder_lin1`, `encoder_lin2`, `encoder_bn5`

### Step 2: Calculate training mean features

```bash
python calculate_training_mean_features.py
```

Computes the mean feature across all training samples (used for scl2vis normalization).

Input: `data/features/train-3d-natural-objects/atlasnet/`

Output: `data/features/train-3d-natural-objects/atlasnet_training_mean/`

## Scripts

| Script | Description |
|---|---|
| `extract_features_atlasnet.py` | Feature extraction with the AtlasNet model |
| `calculate_training_mean_features.py` | Compute per-layer mean features over training set |
