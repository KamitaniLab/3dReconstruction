from typing import Optional

import os
from glob import glob
from itertools import product

from bdpy.dl.torch import ReconstructionBase
from bdpy.dataform import Features, DecodedFeatures
import numpy as np
import torch
from tqdm import tqdm

from models.autoencoder import AutoEncoder
from utils.misc import seed_all


CUDA = 'cuda:0'


class DiffusionAEReconFromFeature(ReconstructionBase):

    def init(self, model_path: str = '.', scale_factor_path: str = '.', source_layer: str = 'shape_latent', n_points: int = 2048) -> None:
        ckpt = torch.load(model_path, map_location=self.device)
        seed_all(ckpt['args'].seed)
        self.model = self.model_cls(ckpt['args']).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.layer = source_layer
        self.n_points = n_points
        self.flexibility = ckpt['args'].flexibility
        self.scale_factor = float(np.load(scale_factor_path))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        '''
        Preprocess an input feature for DiffusionPC reconstruction.
        '''
        x = x.squeeze()[np.newaxis]
        return torch.Tensor(x)

    def reconstruct(self, x: torch.Tensor) -> np.ndarray:
        '''
        Reconstruct a point cloud from the input latent feature.
        Output is rescaled back to AtlasNet scale (divided by scale_factor).
        '''
        x = x.to(self.device)

        with torch.no_grad():
            pc = self.model.decode(x, self.n_points, flexibility=self.flexibility)

        pc = pc.cpu().numpy().squeeze(0)  # (1, N, 3) -> (N, 3)
        pc = pc / self.scale_factor       # DiffusionPC scale (std~1) -> AtlasNet scale
        return pc


def reconstruct_all(recon, features, output_dir, source_layer, subject=None, roi=None):
    os.makedirs(os.path.join(output_dir, 'pointcloud'), exist_ok=True)

    for label in tqdm(features.labels):
        print(label)

        try:
            if subject is not None:
                feat = features.get(layer=source_layer, subject=subject, roi=roi, label=label)
            else:
                feat = features.get(layer=source_layer, label=label)
        except Exception:
            print('Feature not found. Skipped.')
            continue

        pc_file = os.path.join(output_dir, 'pointcloud', label + '.npy')
        if not os.path.exists(pc_file):
            pc = recon(feat)
            np.save(pc_file, pc)
            print(f'Saved {pc_file}')


if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    output_root_dir = os.path.join(data_dir, 'reconstruction')

    models = [
        {
            'name':   'diffusion_point_cloud',
            'path':   os.path.join(data_dir, 'models', 'diffusion_point_cloud', 'ckpt.pt'),
            'layer':  'shape_latent',
        },
    ]

    scale_factor_path = os.path.join(data_dir, 'models', 'diffusion_point_cloud', 'scaling_factor.npy')

    # True features
    true_datasets = [
        'test-3d-natural-objects',
    ]

    # Decoded features
    decoded_datasets = [
        'train-3d-natural-objects_rep3_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
    ]

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
    rois = ['WholeVC', 'EarlyVC', 'MTVC', 'DorsalVC', 'VentralVC']

    # Reconstruction from true features
    for dataset, model in product(true_datasets, models):
        model_name   = model['name']
        model_path   = model['path']
        source_layer = model['layer']

        feature_path = os.path.join(data_dir, 'features', dataset, model_name)
        if not os.path.exists(feature_path):
            print(f'{feature_path} does not exist. Skipped.')
            continue

        print(dataset)
        print(model_name)

        output_dir = os.path.join(output_root_dir, f'{model_name}_{source_layer}', 'true', dataset)
        features = Features(feature_path)

        recon = DiffusionAEReconFromFeature(
            model_cls=AutoEncoder,
            device=CUDA,
            init_args={
                'model_path':        model_path,
                'scale_factor_path': scale_factor_path,
                'source_layer':      source_layer,
            }
        )

        reconstruct_all(recon, features, output_dir, source_layer)

    # Reconstruction from decoded features
    for exp, model, sub, roi in product(decoded_datasets, models, subjects, rois):
        model_name   = model['name']
        model_path   = model['path']
        source_layer = model['layer']

        feature_path = os.path.join(data_dir, 'decoded-features', exp, model_name)
        if not os.path.exists(feature_path):
            print(f'{feature_path} does not exist. Skipped.')
            continue

        features = DecodedFeatures(feature_path)

        if not (sub in features.subjects and roi in features.rois):
            print(f'{sub} and {roi} not found. Skipped.')
            continue

        print(exp)
        print(model_name)
        print(f'{sub} - {roi}')

        output_dir = os.path.join(output_root_dir, f'{model_name}_{source_layer}', 'decoded', exp, sub, roi)

        recon = DiffusionAEReconFromFeature(
            model_cls=AutoEncoder,
            device=CUDA,
            init_args={
                'model_path':        model_path,
                'scale_factor_path': scale_factor_path,
                'source_layer':      source_layer,
            }
        )

        reconstruct_all(recon, features, output_dir, source_layer, subject=sub, roi=roi)
