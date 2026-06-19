"""Metrics for evaluating 3D representations and reconstruction."""

from math import comb
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def target_distances_from_distance_matrix(distance_matrix: pd.DataFrame) -> dict[str, float]:
    """Return the target distance for each row in a labeled distance matrix."""
    distances = distance_matrix.to_numpy()
    row_labels = distance_matrix.index.to_numpy()
    col_labels = distance_matrix.columns.to_numpy()
    label_to_col = {label: index for index, label in enumerate(col_labels)}
    missing_labels = [label for label in row_labels if label not in label_to_col]
    if missing_labels:
        preview = ", ".join(map(str, missing_labels[:5]))
        raise ValueError(f"Missing target labels in distance-matrix columns: {preview}")

    col_index = np.asarray([label_to_col[label] for label in row_labels])
    target_distances = distances[np.arange(distances.shape[0]), col_index]
    return dict(zip(row_labels, target_distances))


def identification_accuracy_from_distance_matrix(
    distance_matrix: pd.DataFrame,
    *,
    num_lures: int = 1,
) -> dict[str, float]:
    """Calculate identification accuracy from a labeled distance matrix.

    Smaller distances indicate better matches. The target for each row is the
    column with the same stimulus label as the row. With the default
    ``num_lures=1``, this is the two-way identification accuracy used in the
    paper.
    """
    distances = distance_matrix.to_numpy()
    row_labels = distance_matrix.index.to_numpy()
    col_labels = distance_matrix.columns.to_numpy()
    label_to_col = {label: index for index, label in enumerate(col_labels)}
    missing_labels = [label for label in row_labels if label not in label_to_col]
    if missing_labels:
        preview = ", ".join(map(str, missing_labels[:5]))
        raise ValueError(f"Missing target labels in distance-matrix columns: {preview}")

    col_index = np.asarray([label_to_col[label] for label in row_labels])
    target_distances = distances[np.arange(distances.shape[0]), col_index]
    lure_mask = np.ones(distances.shape, dtype=bool)
    lure_mask[np.arange(distances.shape[0]), col_index] = False

    good_lures_mask = (distances > target_distances[:, None]) & lure_mask
    rank_counts = good_lures_mask.sum(axis=1)
    total_lures = lure_mask.sum(axis=1)
    lures_per_trial = num_lures if num_lures != np.inf else total_lures[0]
    denom = comb(total_lures[0], lures_per_trial)

    def trial_probability(rank_count: int) -> float:
        return comb(rank_count, lures_per_trial) / denom if rank_count >= lures_per_trial else 0.0

    accuracies = np.fromiter(
        (trial_probability(rank_count) for rank_count in rank_counts),
        dtype=float,
        count=distances.shape[0],
    )
    return dict(zip(row_labels, accuracies))


def chamfer_distance(in_cham1: torch.Tensor, in_cham2: torch.Tensor) -> torch.Tensor:
    """Calculate chamfer distance between two point clouds.

    Parameters
    ----------
    in_cham1 : torch.Tensor
        First point cloud tensor.
    in_cham2 : torch.Tensor
        Second point cloud tensor.

    Returns
    -------
    torch.Tensor
        Chamfer distance between the two point clouds.
    """
    in_cham1 = in_cham1.contiguous().view(in_cham1.size(0), -1, 3).contiguous()
    in_cham2 = in_cham2.contiguous().view(in_cham2.size(0), -1, 3).contiguous()
    dist1, dist2, idx1, idx2 = dist_chamfer(in_cham1, in_cham2)
    chamfer_per_batch = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    return chamfer_per_batch.cpu()


def dist_chamfer(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate chamfer distance between two point clouds.

    Parameters
    ----------
    a : torch.Tensor
        Pointclouds Batch x nul_points x dim.
    b : torch.Tensor
        Pointclouds Batch x nul_points x dim.

    Returns
    -------
    tuple
        A tuple containing:
        - torch.Tensor: closest point on b of points from a
        - torch.Tensor: closest point on a of points from b
        - torch.Tensor: idx of closest point on b of points from a
        - torch.Tensor: idx of closest point on a of points from b

    Notes
    -----
    Works for pointcloud of any dimension.
    """
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x)  # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)  # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()
