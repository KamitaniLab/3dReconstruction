"""Contour-matched slant analysis data loading."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from recon3d.metadata import (
    CONTOUR_SLANT_ROIS,
    CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT,
    CONTOUR_SLANT_TRIALS_INCLUDING_LARGEST_SLANT,
    PUBLIC_SUBJECTS,
)


HORIZONTAL_SHAPE_DATASET = "test-3d-contour-matched-rds-horizontal-shape-variants"
THIN_TILT_DATASET = "test-3d-contour-matched-rds-thin-tilt-variants"
TRAIN_DATASET = "train-3d-natural-objects_rep3"
FMRI_SUFFIX = "rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000"

ROI_REF = "EarlyVC"

STIMULI = [
    "horizontal_thin_bar",
    "horizontal_thick_bar",
    "horizontal_cylinder",
    "vertical_thin_bar",
]
HV_STIMULI = ["horizontal_thin_bar", "vertical_thin_bar"]

HV_INDEX_DIR = {
    "horizontal_thin_bar": (2, 0, -1, 1),
    "horizontal_thick_bar": (2, 0, -1, 1),
    "horizontal_cylinder": (2, 0, -1, 1),
    "vertical_thin_bar": (1, 0, -1, 1),
}


def reconstruction_run_name(test_dataset: str) -> str:
    return f"{TRAIN_DATASET}_{test_dataset}_{FMRI_SUFFIX}"


def normalize_angle(angle: float | np.ndarray) -> np.ndarray:
    """Normalize degrees to [-90, 90)."""
    return (np.asarray(angle, dtype=float) + 90) % 180 - 90


def eigen_vector_to_angle(
    eigen_vector: np.ndarray,
    *,
    h_index: int,
    v_index: int,
    h_dir: int,
    v_dir: int,
) -> float:
    v2d = np.array([h_dir * eigen_vector[h_index], v_dir * eigen_vector[v_index]])
    angle = np.rad2deg(math.atan2(v2d[1], v2d[0]))
    if angle < 0:
        angle = 180 + angle
    if angle > 180:
        angle = angle - 180
    return float(normalize_angle(np.array([angle]))[0])


@dataclass
class ShapeData:
    pointcloud_file: Path
    pointcloud: np.ndarray = field(init=False)
    eigen_vector: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.pointcloud = np.load(self.pointcloud_file).squeeze()
        self.eigen_vector = self._first_principal_component(self.pointcloud)

    @staticmethod
    def _first_principal_component(pointcloud: np.ndarray) -> np.ndarray:
        centered = pointcloud - np.mean(pointcloud, axis=0)
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, order[0]]


def _reconstructed_shapes(path: Path) -> dict[str, ShapeData]:
    shapes = {}
    for file in sorted(path.glob("*.npy")):
        if file.name.startswith("image"):
            continue
        shapes[file.stem] = ShapeData(file)
    if not shapes:
        raise FileNotFoundError(f"No reconstructed point-cloud files found in {path}")
    return shapes


def _reconstruction_dir(data_root: Path, model_name: str, test_dataset: str) -> Path:
    return data_root / "reconstruction" / model_name / "decoded" / reconstruction_run_name(test_dataset)


def build_contour_slant_dataframe(
    *,
    data_root: Path,
    model_name: str,
    subjects: list[str] | None = None,
    rois: list[str] | None = None,
    stimuli: list[str] | None = None,
    use_thin_tilt_horizontal: bool = False,
    include_largest_slant: bool = False,
) -> pd.DataFrame:
    """Build the LMM table used for the contour-matched slant analyses."""
    subjects = subjects or PUBLIC_SUBJECTS
    rois = rois or CONTOUR_SLANT_ROIS
    stimuli = stimuli or STIMULI
    trial_table = (
        CONTOUR_SLANT_TRIALS_INCLUDING_LARGEST_SLANT
        if include_largest_slant
        else CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT
    )

    rows = []
    for subject, roi, stimulus in product(subjects, rois, stimuli):
        if use_thin_tilt_horizontal and stimulus == "horizontal_thin_bar":
            test_dataset = THIN_TILT_DATASET
        elif stimulus == "vertical_thin_bar":
            test_dataset = THIN_TILT_DATASET
        else:
            test_dataset = HORIZONTAL_SHAPE_DATASET
        recon_dir = _reconstruction_dir(data_root, model_name, test_dataset) / subject / roi / "pointcloud"
        recon_shapes = _reconstructed_shapes(recon_dir)
        recon_keys = list(recon_shapes)
        h_index, v_index, h_dir, v_dir = HV_INDEX_DIR[stimulus]

        trials = trial_table[stimulus]
        stimulus_indices = [int(trial["stimulus_index"]) for trial in trials]
        true_degs = np.asarray([float(trial["exp_true_deg"]) for trial in trials], dtype=float)
        nominal_degs = np.asarray([float(trial["nominal_deg"]) for trial in trials], dtype=float)
        recon_keys_for_trials = [recon_keys[stim_index] for stim_index in stimulus_indices]
        recon_angles = []
        for recon_key in recon_keys_for_trials:
            recon_angle = eigen_vector_to_angle(
                recon_shapes[recon_key].eigen_vector,
                h_index=h_index,
                v_index=v_index,
                h_dir=h_dir,
                v_dir=v_dir,
            )
            recon_angles.append(recon_angle)

        recon_angles = np.asarray(recon_angles, dtype=float)
        pred_degs = normalize_angle(nominal_degs + normalize_angle(recon_angles - nominal_degs))

        for recon_key, true_deg, pred_deg in zip(recon_keys_for_trials, true_degs, pred_degs):
            rows.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "stimulus": stimulus,
                    "test_dataset": test_dataset,
                    "stimulus_file": recon_key,
                    "true_deg": float(true_deg),
                    "pred_deg": pred_deg,
                }
            )

    df = pd.DataFrame(rows)
    df["subject"] = pd.Categorical(df["subject"], categories=subjects).remove_unused_categories()
    df["roi"] = pd.Categorical(df["roi"], categories=rois).remove_unused_categories()
    df["stimulus"] = pd.Categorical(df["stimulus"], categories=stimuli).remove_unused_categories()
    return df
