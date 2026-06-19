# Reconstruction of cue-invariant 3D appearances from brain activity

## Setup

```bash
git submodule update --init --recursive
```

## Steps

1. **Feature extraction** — Extract DNN features from 3D point clouds using AtlasNet.
   See [feature-extraction/atlasnet/README.md](feature-extraction/atlasnet/README.md) for instructions.

2. **Feature decoding** — Decode DNN features from fMRI data using linear decoders.
   See [feature-decoding/README.md](feature-decoding/README.md) for instructions.

3. **Reconstruction** — Reconstruct 3D shapes from true or decoded features using AtlasNet.
   See [reconstruction/atlasnet/README.md](reconstruction/atlasnet/README.md) for instructions.

