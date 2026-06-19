# Feature Extraction

Scripts for extracting DNN features from 3D point clouds using the diffusion-based point-cloud autoencoder.

## Setup

### Python environment

This directory uses [uv](https://docs.astral.sh/uv/) to manage the Python environment.
For exact reproduction of the released features, install dependencies from the included `uv.lock`.

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

Model weights and scaling factor must be placed at:

```
data/models/diffusion_point_cloud/
├── ckpt.pt
└── scaling_factor.npy
```

Scripts can be run from any directory (paths are resolved from `__file__`).

## Usage

### Step 1: Calculate scaling factor

```bash
uv run save_scaling_factor.py
```

Computes the scale factor from the training point clouds.

This step is specific to the diffusion-based point-cloud autoencoder. The point clouds in this repository are shared across multiple 3D DNNs, but the DNNs were trained with different normalization rules. AtlasNet normalizes each point cloud into a unit sphere, so that the object fits inside a sphere with radius 1. In contrast, the diffusion-based point-cloud autoencoder was trained on point clouds whose coordinates have approximately zero mean and unit standard deviation.

Because of this difference, the same point-cloud files cannot be passed directly to both DNNs without adjusting their numerical scale. Before extracting diffusion-based point-cloud autoencoder features, the point clouds must be multiplied by a constant scale factor so that their coordinate scale matches the scale used during diffusion-based point-cloud autoencoder training. This is similar to applying the correct image normalization before extracting features from an image DNN: the stimulus is the same, but the numerical input range expected by the DNN is different.

The script estimates this constant using only the training set. It computes the standard deviation of each training point cloud, averages those values, and saves the inverse of that average as the scaling factor. The saved value is then loaded by `extract_features_diffusion_point_cloud.py` and applied to every point cloud before feature extraction.

Output: `data/models/diffusion_point_cloud/scaling_factor.npy`

### Step 2: Extract features

```bash
uv run extract_features_diffusion_point_cloud.py
```

Extracts encoder features from point clouds for all datasets.
Already-existing output directories are skipped.

Output: `data/features/{dataset}/diffusion_point_cloud/{layer}/{label}.mat`

| Dataset | Output directory |
|---|---|
| Train 3D Natural Objects | `data/features/train-3d-natural-objects/diffusion_point_cloud/` |
| Test 3D Natural Objects | `data/features/test-3d-natural-objects/diffusion_point_cloud/` |
| Test 3D Artificial Objects | `data/features/test-3d-artificial-objects/diffusion_point_cloud/` |

Extracted layer: `shape_latent`

## Scripts

| Script | Description |
|---|---|
| `save_scaling_factor.py` | Compute the point-cloud scale factor from the training set |
| `extract_features_diffusion_point_cloud.py` | Feature extraction with the diffusion-based point-cloud autoencoder |
