"""Metrics for evaluating 3D reconstruction."""

import os
import pickle
from glob import glob
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch


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


def calc_chamfer_dist_matrix(
    recon_pcs: Dict[str, np.ndarray],
    gt_pcs: Dict[str, np.ndarray],
    device: str = 'cuda:0'
) -> np.ndarray:
    """Calculate chamfer distance matrix between reconstructed and ground truth point clouds.

    Parameters
    ----------
    recon_pcs : Dict[str, np.ndarray]
        Dictionary of reconstructed point clouds.
    gt_pcs : Dict[str, np.ndarray]
        Dictionary of ground truth point clouds.
    device : str, optional
        Device to run calculations on, by default 'cuda:0'.

    Returns
    -------
    np.ndarray
        Matrix of chamfer distances between reconstructed and ground truth point clouds.
    """
    cd_mat = np.zeros((len(recon_pcs), len(gt_pcs)))
    for i, stim_i in enumerate(recon_pcs):
        for j, stim_j in enumerate(gt_pcs):
            rc_pc = recon_pcs[stim_i][np.newaxis, :, :]
            gt_pc = gt_pcs[stim_j][np.newaxis, :, :]
            # rc_pc = (rc_pc - np.mean(rc_pc, axis=1, keepdims=True)) / np.std(rc_pc, axis=1, keepdims=True)
            # gt_pc = (gt_pc - np.mean(gt_pc, axis=1, keepdims=True)) / np.std(gt_pc, axis=1, keepdims=True)
            cd_mat[i, j] = chamfer_distance(
                torch.tensor(gt_pc).to(device),
                torch.tensor(rc_pc).to(device)
            ).detach().clone()
    return cd_mat


def f_score(pred_pc: torch.Tensor, gt_pc: torch.Tensor, threshold_percent: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate F-score between predicted and ground truth point clouds.

    Parameters
    ----------
    pred_pc : torch.Tensor
        Predicted point cloud tensor with shape (batch_size, num_points, 3).
    gt_pc : torch.Tensor
        Ground truth point cloud tensor with shape (batch_size, num_points, 3).
    threshold_percent : float, optional
        Percentage of the side length of the reconstructed volume to use as the threshold, by default 0.01.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - torch.Tensor: F-score values for each batch
        - torch.Tensor: Precision values for each batch
        - torch.Tensor: Recall values for each batch
    """
    pred_pc = pred_pc.contiguous().view(pred_pc.size(0), -1, 3)
    gt_pc = gt_pc.contiguous().view(gt_pc.size(0), -1, 3)
    threshold = fscore_threshold(pred_pc, threshold_percent)

    # Calculate distances
    dist_pred_to_gt, dist_gt_to_pred, _, _ = dist_chamfer(pred_pc, gt_pc)
    dist_pred_to_gt = dist_pred_to_gt.sqrt()
    dist_gt_to_pred = dist_gt_to_pred.sqrt()

    # Calculate precision (how many predicted points are close to ground truth)
    precision = torch.mean((dist_pred_to_gt < threshold).float(), dim=1)

    # Calculate recall (how many ground truth points are close to predictions)
    recall = torch.mean((dist_gt_to_pred < threshold).float(), dim=1)

    # Calculate F-score
    eps = 1e-8  # Small epsilon to avoid division by zero
    f_score_values = 100 *  2 * (precision * recall) / (precision + recall + eps)

    return f_score_values.cpu(), precision.cpu(), recall.cpu()

def fscore_threshold(pred_pc, percent=0.01):
    """Calculate threshold for F-score based on the side length of the reconstructed volume
    following the method in Tatarchenko et al. 2019 "What Do Single-view 3D Reconstruction Networks Learn?".
    """
    pred_pc = pred_pc.contiguous().view(pred_pc.size(0), -1, 3)
    pred_pc_min = pred_pc.min(dim=1)[0]
    pred_pc_max = pred_pc.max(dim=1)[0]
    side_length = pred_pc_max - pred_pc_min
    # Take the maximum side length to get a scalar threshold per batch
    threshold = torch.max(side_length, dim=1)[0] * percent
    return threshold.unsqueeze(1)  # Shape: (batch_size, 1) for broadcasting



def calc_f_score_matrix(
    recon_pcs: Dict[str, np.ndarray],
    gt_pcs: Dict[str, np.ndarray],
    threshold_percent: float = 0.01,
    device: str = 'cuda:0'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate F-score matrix between reconstructed and ground truth point clouds.

    Parameters
    ----------
    recon_pcs : Dict[str, np.ndarray]
        Dictionary of reconstructed point clouds.
    gt_pcs : Dict[str, np.ndarray]
        Dictionary of ground truth point clouds.
    threshold_percent : float, optional
        Percentage of the side length of the reconstructed volume to use as the threshold, by default 0.01.
    device : str, optional
        Device to run calculations on, by default 'cuda:0'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - np.ndarray: Matrix of F-scores
        - np.ndarray: Matrix of precision values
        - np.ndarray: Matrix of recall values
    """
    f_score_mat = np.zeros((len(recon_pcs), len(gt_pcs)))
    precision_mat = np.zeros((len(recon_pcs), len(gt_pcs)))
    recall_mat = np.zeros((len(recon_pcs), len(gt_pcs)))

    for i, stim_i in enumerate(recon_pcs):
        for j, stim_j in enumerate(gt_pcs):
            rc_pc = recon_pcs[stim_i][np.newaxis, :, :]
            gt_pc = gt_pcs[stim_j][np.newaxis, :, :]

            f_score_val, precision_val, recall_val = f_score(
                torch.tensor(rc_pc).to(device),
                torch.tensor(gt_pc).to(device),
                threshold_percent=threshold_percent
            )

            f_score_mat[i, j] = f_score_val.item()
            precision_mat[i, j] = precision_val.item()
            recall_mat[i, j] = recall_val.item()

    return f_score_mat, precision_mat, recall_mat
