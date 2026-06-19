# Data

Data directory for the 3D reconstruction project. It contains fMRI data, DNN features, trained feature decoders, decoded features, network models, ground-truth point clouds, and reconstruction results.

## Datasets

- fMRI data
    - Train 3D Natural Objects (image presentation)
    - Test 3D Natural Objects (image presentation)
    - Test 3D Artificial Objects: Image
    - Test 3D Artificial Objects: RDS (random-dot stereogram)
    - Test Contour-matched RDS: horizontal shape variants
    - Test Contour-matched RDS: thin tilt variants

## Directory structure

```
data/
├── fmri/                          # fMRI data (.h5)
│   └── {subject}_{dataset}_rep{n}_fmap_volume_native_visualcortex.h5
│       e.g. S1_train-3d-natural-objects-image_rep3_fmap_volume_native_visualcortex.h5
│            S1_test-3d-natural-objects-image_rep8_fmap_volume_native_visualcortex.h5
│            S1_test-3d-artificial-objects-image_rep8_fmap_volume_native_visualcortex.h5
│            S1_test-3d-artificial-objects-rds_rep8_fmap_volume_native_visualcortex.h5
│            S1_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_volume_native_visualcortex.h5
│            S1_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_volume_native_visualcortex.h5
├── features/                      # DNN features
│   ├── train-3d-natural-objects/
│   │   └── atlasnet/encoder_bn5/
│   ├── test-3d-natural-objects/
│   │   └── atlasnet/encoder_bn5/
│   ├── test-3d-artificial-objects/
│   │   └── atlasnet/encoder_bn5/
│   ├── test-3d-contour-matched-rds-horizontal-shape-variants/
│   │   └── atlasnet/encoder_bn5/
│   └── test-3d-contour-matched-rds-thin-tilt-variants/
│       └── atlasnet/encoder_bn5/
├── feature-decoders/              # Trained feature decoders
│   └── train-3d-natural-objects-image_rep3_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
├── decoded-features/              # Decoded features
│   ├── train-3d-natural-objects-image_rep3_test-3d-natural-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
│   ├── train-3d-natural-objects-image_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
│   ├── train-3d-natural-objects-image_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
│   ├── train-3d-natural-objects-image_rep3_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
│   └── train-3d-natural-objects-image_rep3_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
├── models/                        # Network models
│   └── atlasnet/network_crtd.pth
├── pointcloud/                    # Ground-truth point clouds (.ply.npy)
│   ├── train-3d-natural-objects/
│   ├── test-3d-natural-objects/
│   ├── test-3d-artificial-objects/
│   ├── test-3d-contour-matched-rds-horizontal-shape-variants/
│   └── test-3d-contour-matched-rds-thin-tilt-variants/
└── reconstruction/                # Reconstruction results
    └── atlasnet_encoder_bn5/
        ├── true/{dataset}/pointcloud/, mesh/
        └── decoded/{exp}/{subject}/{roi}/pointcloud/, mesh/
```

## Downloading data

Use `download.py` with the targets defined in `files.json`:

```
python download.py fmri_visualcortex
python download.py features
```

Available targets in `files.json`:

- `fmri_visualcortex` — visual-cortex-masked fMRI files (`*_visualcortex.h5`) for S1–S5 across all datasets, saved under `./fmri/`.
- `features` — AtlasNet `encoder_bn5` features for the train/test natural and artificial object datasets, saved under `./` and unzipped into `./features/`.

Each file entry includes an MD5 checksum that is verified on download.

## File-naming conventions

### fMRI

```
{subject}_{dataset}_rep{n}_fmap_volume_native_visualcortex.h5
```

- `subject`: `S1`–`S5`
- `dataset`: e.g. `train-3d-natural-objects-image`, `test-3d-natural-objects-image`, `test-3d-artificial-objects-image`, `test-3d-artificial-objects-rds`, `test-3d-contour-matched-rds-horizontal-shape-variants`, `test-3d-contour-matched-rds-thin-tilt-variants`
- `rep{n}`: number of repetitions averaged (`rep3` for training, `rep8` for test)
- `fmap_volume_native_visualcortex`: fieldmap-corrected volume in native space
- `visualcortex`: masked to visual cortex only

### Features

TBA

### Decoder / decoded-feature directories

```
{train_dataset}_{test_dataset}_fmap_fmriprep_5000voxel_fastl2lir_alpha5000/
```

- `fmap`: field map correction applied
- `fmriprep`: preprocessing pipeline
- `5000voxel`: number of voxels selected per ROI
- `fastl2lir_alpha5000`: FastL2LiR decoder with regularization parameter α = 5000

## ROI list

| Key | ROI label |
|---|---|
| EarlyVC | hcp180_EarlyVC |
| VentralVC | hcp180_VentralVC |
| DorsalVC | hcp180_DorsalVC |
| MTVC | hcp180_MTVC |
| WholeVC | hcp180_hcpVC |

Each ROI uses 5000 voxels.

## Network and feature layer

- `network`: `atlasnet`
- `layers`: `encoder_bn5`
- Pretrained weights: `models/atlasnet/network_crtd.pth`
