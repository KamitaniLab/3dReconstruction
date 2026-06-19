# Diffusion-Based Point-Cloud Autoencoder 3D Reconstruction

Scripts for reconstructing 3D shapes (point cloud) from DNN features using the diffusion-based point-cloud autoencoder.

## Setup

Run scripts from the `reconstruction/diffusion-point-cloud/` directory.

```bash
cd reconstruction/diffusion-point-cloud
```

This directory uses [uv](https://docs.astral.sh/uv/) to manage the Python environment.
For exact reproduction of the released reconstructions, install dependencies from the included `uv.lock`.

```bash
uv sync
```

The autoencoder weights and scaling factor must be placed at:

```
data/models/diffusion_point_cloud/
├── ckpt.pt
└── scaling_factor.npy
```

## Usage

```bash
uv run recon_from_features.py
```

Both true-feature reconstruction and decoded-feature reconstruction are executed in a single run.

Already-saved results are skipped (not overwritten).

## Output

Results are saved under `data/reconstruction/diffusion_point_cloud_shape_latent/`.

The diffusion-based point-cloud autoencoder was trained with point clouds normalized to zero mean and unit standard deviation. During reconstruction, the decoded point clouds are rescaled back using `scaling_factor.npy` before being saved. Therefore, the saved point clouds are in the same scale as the point clouds used by AtlasNet-based analyses.

### From true features

Input: `data/features/{dataset}/diffusion_point_cloud/`

Output: `data/reconstruction/diffusion_point_cloud_shape_latent/true/{dataset}/`

```
true/
└── test-3d-natural-objects/
    └── pointcloud/
        └── {label}.npy
```

### From decoded features

Input: `data/decoded-features/{experiment}/diffusion_point_cloud/`

Output: `data/reconstruction/diffusion_point_cloud_shape_latent/decoded/{experiment}/{subject}/{roi}/`

```
decoded/
└── {experiment}/
    └── {subject}/        # S1-S5
        └── {roi}/         # WholeVC, EarlyVC, MTVC, DorsalVC, VentralVC
            └── pointcloud/
                └── {label}.npy
```

Experiments:

| Experiment | Test data |
|---|---|
| `train-3d-natural-objects_rep3_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Test Natural Objects |
| `train-3d-natural-objects_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Artificial Objects: Image |
| `train-3d-natural-objects_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Artificial Objects: RDS |
