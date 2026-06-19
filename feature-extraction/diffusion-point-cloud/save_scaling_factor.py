import os
from glob import glob

import numpy as np


# Main #######################################################################

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    # Point clouds used to compute the scaling factor.
    # Must be the same set used for AtlasNet training (scl2vis_scaledx1p8 scale).
    pc_files = sorted(glob(os.path.join(data_dir, 'pointcloud', 'train-3d-natural-objects', '*.npy')))

    if len(pc_files) == 0:
        raise FileNotFoundError(
            f'No .npy files found in {os.path.join(data_dir, "pointcloud", "train-3d-natural-objects")}'
        )

    # Compute mean std across all training objects
    stds = []
    for pc_file in pc_files:
        pc = np.load(pc_file)   # (N, 3)
        stds.append(pc.std())
    mean_std = np.mean(stds)

    # scale_factor rescales point clouds so that std ~ 1 (DiffusionPC training scale)
    scale_factor = 1.0 / mean_std
    print(f'mean_std={mean_std:.6f}, scale_factor={scale_factor:.10f}')

    output_path = os.path.join(data_dir, 'models', 'diffusion_point_cloud', 'scaling_factor.npy')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, scale_factor)
    print(f'Saved to {output_path}')
