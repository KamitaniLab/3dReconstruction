"""Feature-decoding evaluation data loading and aggregation."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from recon3d.metrics import (
    identification_accuracy_from_distance_matrix,
    target_distances_from_distance_matrix,
)
from recon3d.metadata import INSAMPLE_CATEGORIES, WHOLE_VISUAL_ROI
from recon3d.stats import calc_ci
from recon3d.subjects import PUBLIC_SUBJECTS


FEATURE_DECODING_STIMULUS_SETS = [
    "test-3d-natural-objects_rep8_insample",
    "test-3d-natural-objects_rep8_outsample",
    "test-3d-artificial-objects-image_rep8",
    "test-3d-artificial-objects-rds_rep8",
]


def calculate_2way_identification_accuracy(
    distance_matrix: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate two-way identification accuracy for each stimulus."""
    target_distances = target_distances_from_distance_matrix(distance_matrix)
    pattern_correlations = {
        label: 1 - distance for label, distance in target_distances.items()
    }
    identification_accuracy = identification_accuracy_from_distance_matrix(
        distance_matrix,
        num_lures=1,
    )
    return pattern_correlations, identification_accuracy


def feature_decoding_result_dir(
    results_root: str | Path,
    representation_name: str,
    test_dataset: str,
    subject: str,
    roi: str,
) -> Path:
    """Return the directory containing feature-decoding CSV outputs."""
    return Path(results_root) / representation_name / test_dataset / subject / roi


def load_correlation_distance_matrix(
    results_root: str | Path,
    representation_name: str,
    test_dataset: str,
    subject: str,
    roi: str,
) -> pd.DataFrame:
    """Load a saved feature-decoding correlation-distance matrix."""
    path = (
        feature_decoding_result_dir(results_root, representation_name, test_dataset, subject, roi)
        / "correlation_distance_matrix.csv"
    )
    return pd.read_csv(path, index_col="stimulus")


def load_profile_correlations(
    results_root: str | Path,
    representation_name: str,
    test_dataset: str,
    subject: str,
    roi: str,
) -> pd.DataFrame:
    """Load saved unit-wise profile correlations."""
    path = (
        feature_decoding_result_dir(results_root, representation_name, test_dataset, subject, roi)
        / "profile_correlation.csv"
    )
    return pd.read_csv(path)


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


def compute_feature_decoding_metric(
    results_root: str | Path,
    representation_name: str,
    stimulus_sets: list[str],
    *,
    metric: str,
    subjects: list[str] = PUBLIC_SUBJECTS,
    rois: list[str] | None = None,
) -> dict:
    """Compute mean and CI values for one feature-decoding metric."""
    allowed_metrics = {"profile correlation", "identification accuracy"}
    if metric not in allowed_metrics:
        raise ValueError(f"Unknown feature-decoding metric: {metric}")

    rois = rois or [WHOLE_VISUAL_ROI]
    results = {}

    for stimulus_set in stimulus_sets:
        test_dataset = stimulus_set.replace("_insample", "").replace("_outsample", "")
        mean = np.zeros((len(rois), len(subjects)))
        ci = np.zeros((len(rois), len(subjects), 2))

        for (roi_index, roi), (subject_index, subject) in product(
            enumerate(rois), enumerate(subjects)
        ):
            distance_matrix = load_correlation_distance_matrix(
                results_root,
                representation_name,
                test_dataset,
                subject,
                roi,
            )
            _, identification_accuracy = calculate_2way_identification_accuracy(distance_matrix)
            labels = stimulus_set_labels(stimulus_set, identification_accuracy)

            profile_corr = load_profile_correlations(
                results_root,
                representation_name,
                test_dataset,
                subject,
                roi,
            )

            values = (
                profile_corr.loc[
                    profile_corr["stimulus_set"] == stimulus_set,
                    "profile_correlation",
                ].to_numpy()
                if metric == "profile correlation"
                else np.asarray([identification_accuracy[label] for label in labels])
            )
            mean[roi_index, subject_index] = values.mean()
            ci[roi_index, subject_index, :] = calc_ci(values)

        results[stimulus_set] = {"mean": mean, "ci": ci}

    return results
