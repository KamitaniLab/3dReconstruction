"""Evaluate feature decoding for public natural and artificial object datasets."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from bdpy.dataform import DecodedFeatures, Features
from bdpy.evals.metrics import profile_correlation
import pandas as pd
from scipy.spatial.distance import cdist

from recon3d.config import REPO_ROOT, load_yaml_config, resolve_path
from recon3d.evaluation.feature_decoding import FEATURE_DECODING_STIMULUS_SETS
from recon3d.metadata import ALL_ROIS, INSAMPLE_CATEGORIES, public_stimulus_label
from recon3d.subjects import PUBLIC_SUBJECTS


TEST_DATASETS = [
    "test-3d-natural-objects_rep8",
    "test-3d-artificial-objects-image_rep8",
    "test-3d-artificial-objects-rds_rep8",
]
LURE_FEATURE_DATASET_BY_TEST_DATASET = {
    "test-3d-natural-objects_rep8": "test-3d-natural-objects",
    "test-3d-artificial-objects-image_rep8": "test-3d-artificial-objects",
    "test-3d-artificial-objects-rds_rep8": "test-3d-artificial-objects",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/atlasnet.yaml"))
    return parser.parse_args()


def decoded_feature_root_dir(
    *,
    root: Path,
    test_dataset: str,
    decoded_feature_name: str,
) -> Path:
    """Return the public decoded-feature directory for one test dataset."""
    test_dataset_name = test_dataset.removeprefix("test-")
    run_dirs = sorted(root.glob(f"*_test-{test_dataset_name}_fmap_*"))
    if len(run_dirs) != 1:
        raise FileNotFoundError(
            f"Expected one decoded-feature run for {test_dataset}; found {len(run_dirs)}."
        )
    return run_dirs[0] / decoded_feature_name


def load_ground_truth_features(
    *,
    feature_root: str | Path,
    feature_name: str,
    source_layer: str,
    feature_dataset: str,
) -> dict[str, np.ndarray]:
    """Load ground-truth DNN features used as identification lures."""
    feature_dir = Path(feature_root) / feature_dataset / feature_name
    features = Features(str(feature_dir))
    labels = list(features.labels)
    values = features.get(layer=source_layer).squeeze()
    return dict(zip(labels, values))


def load_decoded_features(
    *,
    decoded_features: DecodedFeatures,
    source_layer: str,
    subject: str,
    roi: str,
) -> dict[str, np.ndarray]:
    """Load decoded DNN features for one subject, ROI, and test dataset."""
    labels = list(decoded_features.labels)
    values = np.concatenate(
        [
            decoded_features.get(
                layer=source_layer,
                subject=subject,
                roi=roi,
                label=label,
            )
            for label in labels
        ],
        axis=0,
    ).squeeze()
    return dict(zip(labels, values))


def calculate_correlation_distance_matrix(
    decoded_features: dict[str, np.ndarray],
    lure_features: dict[str, np.ndarray],
) -> np.ndarray:
    """Calculate correlation distances between decoded and ground-truth features."""
    decoded_values = np.asarray(list(decoded_features.values()))
    lure_values = np.asarray(list(lure_features.values()))
    return cdist(decoded_values, lure_values, metric="correlation")


def target_label(label: str) -> str:
    """Return the ground-truth label corresponding to a decoded-feature label."""
    return public_stimulus_label(label)


def stimulus_set_labels(
    stimulus_set: str,
    labels: list[str],
) -> list[str]:
    """Return the stimulus labels used for one feature-decoding stimulus set."""
    test_dataset = stimulus_set.replace("_insample", "").replace("_outsample", "")
    if test_dataset == "test-3d-natural-objects_rep8" and "_insample" in stimulus_set:
        return [label for label in labels if label.split("_")[2] in INSAMPLE_CATEGORIES]
    if test_dataset == "test-3d-natural-objects_rep8" and "_outsample" in stimulus_set:
        return [label for label in labels if label.split("_")[2] not in INSAMPLE_CATEGORIES]
    return labels


def profile_correlation_table(
    *,
    true_features: np.ndarray,
    decoded_features: np.ndarray,
    labels: list[str],
    test_dataset: str,
) -> pd.DataFrame:
    """Calculate unit-wise profile correlations for stimulus sets in one test dataset."""
    rows = []
    labels_array = np.asarray(labels)
    for stimulus_set in FEATURE_DECODING_STIMULUS_SETS:
        stimulus_set_test_dataset = stimulus_set.replace("_insample", "").replace("_outsample", "")
        if stimulus_set_test_dataset != test_dataset:
            continue

        use_labels = stimulus_set_labels(stimulus_set, labels)
        use_index = np.array(
            [index for index, label in enumerate(labels_array) if label in use_labels],
            dtype=int,
        )
        correlations = profile_correlation(
            true_features[use_index],
            decoded_features[use_index],
        ).flatten()
        rows.extend(
            {
                "stimulus_set": stimulus_set,
                "unit": unit_index,
                "profile_correlation": correlation,
            }
            for unit_index, correlation in enumerate(correlations)
        )
    return pd.DataFrame(rows)


def save_feature_decoding_csvs(
    config: dict,
    *,
    feature_root: str | Path | None = None,
    decoded_feature_root: str | Path | None = None,
    results_root: str | Path | None = None,
) -> Path:
    """Evaluate feature decoding and save CSV files."""
    representation = config["representation"]
    feature_root = Path(feature_root or REPO_ROOT / "data" / "features")
    decoded_feature_root = Path(decoded_feature_root or REPO_ROOT / "data" / "decoded-features")
    results_root = Path(
        results_root
        or resolve_path(config["feature_decoding"]["results_dir"], base=Path.cwd())
    )
    representation_results_root = results_root / representation["name"]

    lure_features_by_test_dataset = {
        test_dataset: load_ground_truth_features(
            feature_root=feature_root,
            feature_name=representation["feature_name"],
            source_layer=representation["source_layer"],
            feature_dataset=feature_dataset,
        )
        for test_dataset, feature_dataset in LURE_FEATURE_DATASET_BY_TEST_DATASET.items()
    }
    decoded_features_by_test_dataset = {
        test_dataset: DecodedFeatures(
            str(
                decoded_feature_root_dir(
                    root=decoded_feature_root,
                    test_dataset=test_dataset,
                    decoded_feature_name=representation["decoded_feature_name"],
                )
            )
        )
        for test_dataset in TEST_DATASETS
    }

    for test_dataset, subject, roi in product(TEST_DATASETS, PUBLIC_SUBJECTS, ALL_ROIS):
        lure_features = lure_features_by_test_dataset[test_dataset]
        decoded_features = load_decoded_features(
            decoded_features=decoded_features_by_test_dataset[test_dataset],
            source_layer=representation["source_layer"],
            subject=subject,
            roi=roi,
        )
        labels = list(decoded_features)
        missing_labels = sorted(
            label for label in labels if target_label(label) not in lure_features
        )
        if missing_labels:
            missing = ", ".join(missing_labels[:5])
            raise KeyError(
                f"{test_dataset} has decoded features without matching lures: {missing}"
            )

        output_dir = representation_results_root / test_dataset / subject / roi
        output_dir.mkdir(parents=True, exist_ok=True)

        distance_matrix = calculate_correlation_distance_matrix(
            decoded_features,
            lure_features,
        )
        output_labels = [target_label(label) for label in labels]
        pd.DataFrame(
            distance_matrix,
            index=output_labels,
            columns=list(lure_features),
        ).to_csv(output_dir / "correlation_distance_matrix.csv", index_label="stimulus")

        true_features = np.asarray([lure_features[target_label(label)] for label in labels])
        decoded_feature_values = np.asarray([decoded_features[label] for label in labels])
        profile_correlation_table(
            true_features=true_features,
            decoded_features=decoded_feature_values,
            labels=labels,
            test_dataset=test_dataset,
        ).to_csv(output_dir / "profile_correlation.csv", index=False)

    return representation_results_root


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    results_dir = save_feature_decoding_csvs(config)
    print(f"Saved: {results_dir}")


if __name__ == "__main__":
    main()
