# Feature Decoding

Scripts for training feature decoders and decoding DNN features from fMRI data using FastL2LiR.

## Setup

All scripts must be run from the `feature-decoding/` directory.

```bash
cd feature-decoding
```

### Environment setup with uv

```bash
uv sync
```

This creates a `.venv` directory with all dependencies installed.

## Usage

Scripts can be run either by activating the virtual environment or using `uv run`.

_Option A: Activate virtual environment_

```bash
source .venv/bin/activate
python scripts/train_decoder_fastl2lir.py <config_file>
python scripts/predict_feature_fastl2lir.py <config_file>
```

_Option B: uv run_

```bash
uv run python scripts/train_decoder_fastl2lir.py <config_file>
uv run python scripts/predict_feature_fastl2lir.py <config_file>
```

### 1. Train feature decoders

Train a linear decoder that maps fMRI responses to DNN features.

```bash
python scripts/train_decoder_fastl2lir.py train-3d-natural-objects_rep3_fmap_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml
```

Output: feature decoders saved to the path specified by `decoder.path` in the config.

### 2. Decode features

Apply trained decoders to test fMRI data to predict DNN features.

```bash
python scripts/predict_feature_fastl2lir.py train-3d-natural-objects_rep3_fmap_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml
```

Output: decoded features saved to the path specified by `decoded_feature.path` in the config.

## Config files

Config files are located in `config/`. Each file specifies a combination of training and test datasets.

| Config file | Training data | Test data |
|---|---|---|
| `train-3d-natural-objects-image_rep3_fmap_test-3d-natural-objects-image_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml` | 3D Natural Objects (train) | 3D Natural Objects (test) |
| `train-3d-natural-objects-image_rep3_fmap_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml` | 3D Natural Objects (train) | 3D Artificial Objects: Image |
| `train-3d-natural-objects-image_rep3_fmap_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml` | 3D Natural Objects (train) | 3D Artificial Objects: RDS |
| `train-3d-natural-objects-image_rep3_fmap_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml` | 3D Natural Objects (train) | Contour-matched RDS: horizontal shape variants |
| `train-3d-natural-objects-image_rep3_fmap_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml` | 3D Natural Objects (train) | Contour-matched RDS: thin tilt variants |

## Config file format

```yaml
decoder:
  path: ../data/feature-decoders/<decoder_name>
  parameters:
    alpha: 5000
    chunk_axis: 1
  fmri:
    subjects:
      - name: S1
        paths:
          - ../data/fmri/S1_train-..._fmap_volume_native_visualcortex.h5
      # S2–S5 follow the same pattern
    rois:
      - {name: EarlyVC, select: hcp180_EarlyVC, num: 5000}
      # ...
    label_key: stimulus_name
  features:
    paths:
      - ../data/features/train-3d-natural-objects/atlasnet
    layers:
      - encoder_bn5

decoded_feature:
  path: ../data/decoded-features/<experiment_name>
  parameters:
    average_sample: true
  decoder:
    path: ../data/feature-decoders/<decoder_name>
  fmri:
    subjects:
      - name: S1
        paths:
          - ../data/fmri/S1_test-..._fmap_volume_native_visualcortex.h5
    rois:
      - {name: EarlyVC, select: hcp180_EarlyVC}
      # ...
    label_key: stimulus_name
    exclude_labels:
  features:
    paths:
      - ../data/features/test-3d-natural-objects/atlasnet
    layers:
      - encoder_bn5
```

## Data paths

All paths in config files are relative to the `feature-decoding/` directory.

| Key | Path |
|---|---|
| fMRI data | `../data/fmri/` |
| DNN features | `../data/features/` |
| Feature decoders | `../data/feature-decoders/` |
| Decoded features | `../data/decoded-features/` |
