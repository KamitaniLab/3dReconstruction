# Analysis

This directory contains the analysis code used to evaluate reconstructed 3D
shapes, fit statistical models, and generate figures from the reconstruction
results.

The code assumes that feature decoding and reconstruction have already been
run, or that the corresponding public data have been downloaded under
`../data/`. See [`../data/README.md`](../data/README.md) for the expected data
layout and download instructions.

## Directory Structure

- `evaluation/`: scripts for feature-decoding and reconstruction metrics.
- `statistics/`: scripts for linear mixed-effects model analyses.
- `visualization/`: scripts for generating analysis figures.
- `configs/`: representation-specific analysis configuration files.
- `src/recon3d/`: shared analysis utilities.
- `outputs/`: generated outputs. This directory is created by the scripts.

## Setup

Run all commands from this directory:

```bash
cd analysis
uv sync
```

The Python environment is defined by `pyproject.toml` and `uv.lock`. Some
statistical analyses also require R packages; the Docker environment described
below provides a reproducible R setup.

## Input Data

The analysis scripts expect the repository-level `data/` directory to contain
the relevant inputs.

## Configurations

Two analysis configurations are provided:

- `configs/atlasnet.yaml`: Reconstructions using AtlasNet autoencoder.
- `configs/diffusion_point_cloud.yaml`: Reconstructions using diffusion-based point-cloud autoencoder.

Most scripts accept `--config` to select the representation:

```bash
uv run evaluation/evaluate_reconstruction.py --config configs/atlasnet.yaml
uv run evaluation/evaluate_reconstruction.py --config configs/diffusion_point_cloud.yaml
```

## Evaluation

Compute feature-decoding evaluation metrics:

```bash
uv run evaluation/evaluate_feature_decoding.py --config configs/atlasnet.yaml
```

Compute reconstruction evaluation metrics:

```bash
uv run evaluation/evaluate_reconstruction.py --config configs/diffusion_point_cloud.yaml
```

Evaluation outputs are written under `outputs/evaluation/`.

## Statistical Analysis

The LMM scripts reproduce the group-level analyses described in the paper.
The main inference is frequentist (`lmerTest` in R, restricted maximum
likelihood, Kenward-Roger small-sample correction). Bayesian LMMs
(`Bambi`/`PyMC`, NUTS sampler) are run as robustness checks and report 95%
highest-density intervals and the posterior probability of the hypothesised
direction.

The paper analyzes the whole visual cortex separately from the four sub-ROIs,
because Whole VC contains the sub-ROIs. Each LMM script therefore fits a
Whole VC model and a joint 4-ROI model. In the joint 4-ROI models, early VC is
the reference ROI.

### Rendered-vs-RDS Reconstruction LMM

Fit the reconstruction-identification LMMs comparing rendered-image and RDS
test stimuli:

```bash
uv run statistics/fit_lmm_rendered-vs-RDS.py --config configs/atlasnet.yaml
```

This script computes object-identification accuracy from the reconstruction
evaluation outputs and fits:

- a Whole VC model: `accuracy ~ stimulus + (stimulus|subject)`
- a joint 4-ROI model: `accuracy ~ ROI * stimulus + (stimulus|subject)`

Here, `stimulus` denotes RDS vs. 2D rendered images, with 2D rendered images as
the baseline. The implementation stores this factor in the `dataset` column.
The Whole-VC stimulus effect is reported in the whole-VC summary tables, and
the regional inference is the ROI x stimulus interaction relative to early VC.

Run a specific backend:

```bash
uv run statistics/fit_lmm_rendered-vs-RDS.py --backend frequentist
uv run statistics/fit_lmm_rendered-vs-RDS.py --backend bayesian
```

By default, outputs are written to
`outputs/statistics/reconstruction_lmm/<representation>/`.

### Contour-Matched Slant LMM

The main contour-matched slant analysis entry point is:

```bash
uv run statistics/fit_lmm_slope.py
```

This script reproduces the contour-matched slant linear mixed-effects model
analyses used in the paper:

- `slope`: stimulus-wise slope estimation for the contour-matched stimuli.
- `horizontal-vertical`: post-hoc comparison of horizontal and vertical
  thin-bar slopes.

For each stimulus in the `slope` analysis, the script fits:

- a Whole VC model:
  `theta_pred ~ theta_stim + (theta_stim|subject)`
- a joint 4-ROI model:
  `theta_pred ~ ROI * theta_stim + (theta_stim|subject)`

The Whole-VC slope and the per-ROI simple slopes are tested one-sided
(slope > 0). The regional inference is the ROI x theta_stim interaction
relative to early VC. In the implementation, the z-scored variables are named
`pred_deg_z` and `true_deg_z`.

For the exploratory `horizontal-vertical` analysis, the script adds an
orientation factor and fits:

- a Whole VC model:
  `theta_pred ~ theta_stim * orientation + (theta_stim|subject)`
- a joint 4-ROI model:
  `theta_pred ~ ROI * theta_stim * orientation + (theta_stim|subject)`

The reported horizontal-minus-vertical effect is the theta_stim x orientation
interaction and is tested one-sided for horizontal > vertical.

Run a specific analysis or backend:

```bash
uv run statistics/fit_lmm_slope.py --analysis slope
uv run statistics/fit_lmm_slope.py --analysis horizontal-vertical
uv run statistics/fit_lmm_slope.py --backend frequentist
uv run statistics/fit_lmm_slope.py --backend bayesian
```

As a supplemental sensitivity analysis, the largest nominal slants (+/-60 deg)
can be included with:

```bash
uv run statistics/fit_lmm_slope.py --include-largest-slant
```

By default, outputs are written to
`outputs/statistics/contour_matched_lmm/<representation>/`. Frequentist model
outputs include fitted model summaries, fixed and random effects, model
statistics, and data used for fitting. Bayesian outputs include posterior
summaries, `idata.nc`, metadata, and sampler diagnostics when available.

Bayesian sampler defaults are `draws=2000`, `tune=5000`,
`target_accept=0.995`, `max_treedepth=15`, and `random_seed=42`.

## Visualization

Generate figures from evaluation and statistical outputs:

```bash
uv run visualization/plot_reconstruction_multiview.py --config configs/atlasnet.yaml
uv run visualization/plot_reconstruction_identification_accuracy.py --config configs/atlasnet.yaml
uv run visualization/plot_feature_decoding_evaluation.py --config configs/atlasnet.yaml
uv run visualization/plot_contour_slant_lmm_scatter.py --config configs/diffusion_point_cloud.yaml
uv run visualization/plot_contour_slant_lmm_forest.py --config configs/diffusion_point_cloud.yaml
```

For the corresponding supplemental scatter plots:

```bash
uv run visualization/plot_contour_slant_lmm_scatter.py --include-largest-slant
```

Visualization outputs are written under `outputs/visualization/`.

## Docker

The Docker image pins R to version `4.3.3` and installs Python dependencies
from `uv.lock`.

```bash
./scripts/docker_shell.sh
```

The script builds the image if it is not available, then starts an interactive
shell with `analysis/` mounted at `/app/analysis` and `data/` mounted
read-only at `/app/data`.

Inside the container, `python` points to the project environment, so scripts
can be run directly:

```bash
python statistics/fit_lmm_slope.py
```

The `uv run ...` commands shown above also work inside the container.

To use a different image name:

```bash
IMAGE=my-analysis-image ./scripts/docker_shell.sh
```

To run a single command instead of opening a shell:

```bash
./scripts/docker_shell.sh python statistics/fit_lmm_slope.py --backend frequentist
```
