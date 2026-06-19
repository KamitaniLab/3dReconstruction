"""Evaluate 3D reconstruction for public natural and artificial object datasets."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recon3d.config import REPO_ROOT, load_yaml_config, resolve_path
from recon3d.metadata import ALL_ROIS, public_stimulus_label
from recon3d.metrics import chamfer_distance
from recon3d.subjects import PUBLIC_SUBJECTS


TEST_DATASETS = [
    "test-3d-natural-objects_rep8",
    "test-3d-artificial-objects-image_rep8",
    "test-3d-artificial-objects-rds_rep8",
]
LURE_POINTCLOUD_DATASET_BY_TEST_DATASET = {
    "test-3d-natural-objects_rep8": "test-3d-natural-objects",
    "test-3d-artificial-objects-image_rep8": "test-3d-artificial-objects",
    "test-3d-artificial-objects-rds_rep8": "test-3d-artificial-objects",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/diffusion_point_cloud.yaml"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def decoded_reconstruction_root_dir(
    *,
    root: Path,
    representation_name: str,
    test_dataset: str,
) -> Path:
    """Return the decoded-reconstruction directory for one test dataset."""
    test_dataset_name = test_dataset.removeprefix("test-")
    decoded_root = root / representation_name / "decoded"
    run_dirs = sorted(decoded_root.glob(f"*_test-{test_dataset_name}_fmap_*"))
    if len(run_dirs) != 1:
        raise FileNotFoundError(
            f"Expected one reconstruction run for {test_dataset}; found {len(run_dirs)}."
        )
    return run_dirs[0]


def load_pointcloud_directory(path: str | Path) -> dict[str, np.ndarray]:
    """Load point-cloud files as a label-to-array mapping."""
    pointclouds = {}
    for pointcloud_file in sorted(Path(path).glob("*.npy")):
        label = pointcloud_file.stem.replace(".points.ply", "")
        if label.startswith("image_"):
            continue
        pointclouds[label] = np.load(pointcloud_file).squeeze()
    if not pointclouds:
        raise FileNotFoundError(f"No point-cloud files found in {path}")
    return pointclouds


def load_lure_pointclouds(
    *,
    pointcloud_root: str | Path,
    pointcloud_dataset: str,
) -> dict[str, np.ndarray]:
    """Load ground-truth point clouds used as lures."""
    pointcloud_dir = Path(pointcloud_root) / pointcloud_dataset
    return load_pointcloud_directory(pointcloud_dir)


def load_reconstructed_pointclouds(
    *,
    reconstruction_run_dir: str | Path,
    subject: str,
    roi: str,
) -> dict[str, np.ndarray]:
    """Load reconstructed point clouds for one subject and ROI."""
    pointcloud_dir = Path(reconstruction_run_dir) / subject / roi / "pointcloud"
    return load_pointcloud_directory(pointcloud_dir)


def calculate_chamfer_distance_matrix(
    reconstructed_pointclouds: dict[str, np.ndarray],
    lure_pointclouds: dict[str, np.ndarray],
    *,
    device: str,
) -> np.ndarray:
    """Calculate a Chamfer-distance matrix."""
    chamfer_distance_matrix = np.zeros((len(reconstructed_pointclouds), len(lure_pointclouds)))

    for row_index, reconstructed_label in enumerate(reconstructed_pointclouds):
        reconstructed = reconstructed_pointclouds[reconstructed_label][np.newaxis, :, :]
        reconstructed_tensor = torch.tensor(reconstructed).to(device)

        for column_index, lure_label in enumerate(lure_pointclouds):
            lure = lure_pointclouds[lure_label][np.newaxis, :, :]
            lure_tensor = torch.tensor(lure).to(device)

            chamfer_distance_matrix[row_index, column_index] = (
                chamfer_distance(lure_tensor, reconstructed_tensor).detach().clone()
            )

    return chamfer_distance_matrix


def target_label(label: str) -> str:
    """Return the ground-truth label corresponding to a reconstructed point-cloud label."""
    return public_stimulus_label(label)


def save_distance_matrix(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    path: Path,
) -> None:
    """Save a distance matrix as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=row_labels, columns=column_labels).to_csv(
        path,
        index_label="stimulus",
    )


def save_reconstruction_evaluation_csvs(
    config: dict,
    *,
    pointcloud_root: str | Path | None = None,
    reconstruction_root: str | Path | None = None,
    results_root: str | Path | None = None,
    device: str = "cuda:0",
) -> Path:
    """Evaluate reconstruction and save CSV distance matrices."""
    representation = config["representation"]
    pointcloud_root = Path(pointcloud_root or REPO_ROOT / "data" / "pointcloud")
    reconstruction_root = Path(reconstruction_root or REPO_ROOT / "data" / "reconstruction")
    results_root = Path(
        results_root
        or resolve_path(config["reconstruction_evaluation"]["results_dir"], base=Path.cwd())
    )
    representation_results_root = results_root / representation["name"]

    lure_pointclouds_by_test_dataset = {
        test_dataset: load_lure_pointclouds(
            pointcloud_root=pointcloud_root,
            pointcloud_dataset=pointcloud_dataset,
        )
        for test_dataset, pointcloud_dataset in LURE_POINTCLOUD_DATASET_BY_TEST_DATASET.items()
    }
    reconstruction_runs_by_test_dataset = {
        test_dataset: decoded_reconstruction_root_dir(
            root=reconstruction_root,
            representation_name=representation["reconstruction_name"],
            test_dataset=test_dataset,
        )
        for test_dataset in TEST_DATASETS
    }

    for test_dataset, subject, roi in product(TEST_DATASETS, PUBLIC_SUBJECTS, ALL_ROIS):
        print(f"Evaluating {representation['name']} {test_dataset} {subject} {roi}")
        lure_pointclouds = lure_pointclouds_by_test_dataset[test_dataset]
        reconstructed_pointclouds = load_reconstructed_pointclouds(
            reconstruction_run_dir=reconstruction_runs_by_test_dataset[test_dataset],
            subject=subject,
            roi=roi,
        )

        missing_labels = sorted(
            label for label in reconstructed_pointclouds if target_label(label) not in lure_pointclouds
        )
        if missing_labels:
            missing = ", ".join(missing_labels[:5])
            raise KeyError(
                f"{test_dataset} has reconstructions without matching lures: {missing}"
            )

        chamfer_distance_matrix = calculate_chamfer_distance_matrix(
            reconstructed_pointclouds,
            lure_pointclouds,
            device=device,
        )
        output_dir = representation_results_root / test_dataset / subject / roi
        row_labels = [target_label(label) for label in reconstructed_pointclouds]
        column_labels = list(lure_pointclouds)
        save_distance_matrix(
            chamfer_distance_matrix,
            row_labels=row_labels,
            column_labels=column_labels,
            path=output_dir / "chamfer_distance_matrix.csv",
        )

    return representation_results_root


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    results_dir = save_reconstruction_evaluation_csvs(config, device=args.device)
    print(f"Saved: {results_dir}")


if __name__ == "__main__":
    main()
