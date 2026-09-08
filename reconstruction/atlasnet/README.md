# AtlasNet 3D Reconstruction

Scripts for reconstructing 3D shapes (point cloud and mesh) from DNN features using AtlasNet autoencoder.

## Setup

### Environment

The environment setup (system packages and Python environment) is identical to `feature-extraction/atlasnet/`. Follow the [setup instructions](../../feature-extraction/atlasnet/README.md#setup) there, then install dependencies for this directory:

```bash
uv sync
```

### Model weights

The AtlasNet model weights must be placed at:

```
data/models/atlasnet/network_crtd.pth
```

## Feature scaling

Before reconstructing shapes from decoded features, scale the decoded test
features using the standard deviation estimated from cross-validation predictions.

First, complete:

1. Regular feature decoder training and feature prediction.
2. Cross-validation decoder training and prediction described in
   [feature-decoding/README.md](../../feature-decoding/README.md#cross-validation-for-feature-scaling).

Then run:

```bash
uv run feature_scaling.py
```

For every experiment listed in `feature_scaling.py`, the scaled features are saved to:

```text
data/decoded-features/{experiment}_scaled_traincvstd/atlasnet/
```

## Usage

```bash
uv run recon_from_features.py
```

Both true-feature reconstruction and decoded-feature reconstruction are executed in a single run.

Already-saved results are skipped (not overwritten).

## Output

Results are saved under `data/reconstruction/atlasnet_encoder_bn5/`.

### From true features

Input: `data/features/{dataset}/atlasnet/`

Output: `data/reconstruction/atlasnet_encoder_bn5/true/{dataset}/`

```
true/
└── test-3d-natural-objects/
    ├── pointcloud/
    │   └── {label}.npy
    └── mesh/
        └── {label}.ply
```

### From decoded features

Input: `data/decoded-features/{experiment}_scaled_traincvstd/atlasnet/`

Output: `data/reconstruction/atlasnet_encoder_bn5/decoded/{experiment}_scaled_traincvstd/{subject}/{roi}/`

`recon_from_features.py` reconstructs shapes only from the scaled decoded-feature
datasets listed in `decoded_datasets`.

```
decoded/
└── {experiment}/
    └── {subject}/        # S1–S5
        └── {roi}/         # WholeVC, EarlyVC, MTVC, DorsalVC, VentralVC
            ├── pointcloud/
            │   └── {label}.npy
            └── mesh/
                └── {label}.ply
```

Experiments:

| Experiment | Test data |
|---|---|
| `train-3d-natural-objects_rep3_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Test Natural Objects |
| `train-3d-natural-objects_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Artificial Objects: Image |
| `train-3d-natural-objects_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000` | 3D Artificial Objects: RDS |
