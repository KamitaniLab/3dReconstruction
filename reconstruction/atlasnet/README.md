# AtlasNet 3D Reconstruction

Scripts for reconstructing 3D shapes (point cloud and mesh) from DNN features using AtlasNet autoencoder.

## Setup

Run scripts from the `reconstruction/atlasnet/` directory.

```bash
cd reconstruction/atlasnet
```

The AtlasNet model weights must be placed at:

```
data/models/atlasnet/network_crtd.pth
```

## Usage

```bash
python recon_from_features.py
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

Input: `data/decoded-features/{experiment}/atlasnet/`

Output: `data/reconstruction/atlasnet_encoder_bn5/decoded/{experiment}/{subject}/{roi}/`

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
