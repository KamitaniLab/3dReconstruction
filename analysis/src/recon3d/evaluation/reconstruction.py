"""Reconstruction-evaluation data loading and aggregation."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from recon3d.metrics import identification_accuracy_from_distance_matrix
from recon3d.metadata import INSAMPLE_CATEGORIES
from recon3d.stats import calc_ci


def reconstruction_result_dir(
    results_root: str | Path,
    representation_name: str,
    test_dataset: str,
    subject: str,
    roi: str,
) -> Path:
    """Return the directory containing reconstruction-evaluation CSV outputs."""
    return Path(results_root) / representation_name / test_dataset / subject / roi


def load_chamfer_distance_matrix(
    results_root: str | Path,
    representation_name: str,
    test_dataset: str,
    subject: str,
    roi: str,
) -> pd.DataFrame:
    """Load a saved reconstruction Chamfer-distance matrix."""
    path = (
        reconstruction_result_dir(results_root, representation_name, test_dataset, subject, roi)
        / "chamfer_distance_matrix.csv"
    )
    return pd.read_csv(path, index_col="stimulus")


def stimulus_set_labels(
    stimulus_set: str,
    values_by_label: dict[str, float],
    insample_categories: set[str] = INSAMPLE_CATEGORIES,
) -> list[str]:
    """Return labels belonging to one public stimulus set."""
    test_dataset = stimulus_set.replace("_insample", "").replace("_outsample", "")
    if test_dataset == "test-3d-natural-objects_rep8" and "_insample" in stimulus_set:
        return [label for label in values_by_label if label.split("_")[2] in insample_categories]
    if test_dataset == "test-3d-natural-objects_rep8" and "_outsample" in stimulus_set:
        return [label for label in values_by_label if label.split("_")[2] not in insample_categories]
    return list(values_by_label)


def compute_identification_accuracy(
    results_root: str | Path,
    representation_name: str,
    stimulus_sets: list[str],
    subjects: list[str],
    rois: list[str],
    num_lures: int = 1,
    insample_categories: set[str] = INSAMPLE_CATEGORIES,
) -> dict:
    """Compute raw, mean, and CI values for each stimulus set, ROI, and subject."""
    results = {}
    for stimulus_set in stimulus_sets:
        test_dataset = stimulus_set.replace("_insample", "").replace("_outsample", "")

        mean_ia = {roi: {} for roi in rois}
        ci_ia = {roi: {} for roi in rois}
        raw_ia = {roi: {} for roi in rois}

        for roi, subject in product(rois, subjects):
            distance_matrix = load_chamfer_distance_matrix(
                results_root,
                representation_name,
                test_dataset,
                subject,
                roi,
            )
            ident_acc = identification_accuracy_from_distance_matrix(
                distance_matrix,
                num_lures=num_lures,
            )
            labels = stimulus_set_labels(stimulus_set, ident_acc, insample_categories)

            values = np.asarray([ident_acc[name] for name in labels])
            raw_ia[roi][subject] = values
            mean_ia[roi][subject] = values.mean()
            ci_ia[roi][subject] = calc_ci(values)

        results[stimulus_set] = {
            "identification accuracy": {
                "mean": mean_ia,
                "ci": ci_ia,
                "raw": raw_ia,
            }
        }
    return results
