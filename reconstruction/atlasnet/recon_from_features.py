"""AtlasNet autoencoder 3D reconstruction from features."""


import os
from glob import glob
from typing import List, Dict
from itertools import product

from bdpy.dl.torch import ReconstructionBase
from bdpy.dataform import Features, DecodedFeatures
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

from model.trainer_model import TrainerModel
import dataset.mesh_processor as mesh_processor


CUDA = 'cuda:0'


class AtlasNetAEReconFromFeature(ReconstructionBase):

    def init(self, model_path: str = '.', source_layer: str = 'encoder_bn5') -> None:

        class Options(object):
            def __init__(self, model_path):
                self.multi_gpu = [0]  # List of GPUs to use
                self.device = None  # Will be set by TrainerModel.build_network()
                self.SVR = False  # SVR flag
                self.lrate = 0.001  # Learning rate
                self.train_only_encoder = False  # Flag for training only encoder
                self.reload_model_path = model_path  # Leave blank if you don't need to load entire model
                self.reload_decoder_path = model_path  # Path to your saved decoder
                self.reload_optimizer_path = ""  # Leave blank if you don't need to load optimizer
                self.bottleneck_size = 1024  # Size of bottleneck in encoder/decoder
                self.num_layers = 2  # Number of layers in decoder
                self.hidden_neurons = 512  # Number of neurons per layer in decoder
                self.nb_primitives = 1  # Number of primitives
                self.template_type = 'SPHERE'  # Type of template ("sphere" or "cube")
                self.dim_template = 3  # Dimension of the template
                self.number_points = 5000  # Number of points in the template
                self.number_points_eval = 2500  # Number of points to sample on the output mesh
                self.remove_all_batchNorms = False  # Remove batch norms from the decoder
                self.activation = 'relu'  # Activation function for the decoder
                self.batch_size = 32  # Batch size
                self.batch_size_test = 32  # Batch size for testing
                self.normalization = 'UnitBall'
                self.run_single_eval = True  # Run only one evaluation
                self.remove_all_batchNorms = False
                #self.sample = True  # Sample latent vectors instead of reading them from a file

        opt = Options(model_path)

        trainer = TrainerModel()
        trainer.opt = opt
        trainer.build_network()

        self.model = trainer.network
        self.model.eval()

        self.layer = source_layer

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        '''
        Preprocess an input feature for AtlasNetAE reconstruction.
        '''
        x = x.squeeze()[np.newaxis]
        return torch.Tensor(x)

    def reconstruct(self, x: torch.Tensor) -> np.ndarray:
        '''
        Reconstruct 3D shapes (point cloud) from the input feature.
        '''

        x = x.to(self.device)
        x = F.relu(x)

        pc_t = self.model.module.decoder.forward(x, train=False)

        batch_size = pc_t.shape[0]
        pc_t = pc_t.transpose(2, 3).contiguous()
        pc_t = pc_t.view(batch_size, -1, 3)

        return pc_t.detach().cpu().numpy()

    def generate_mesh(self, x: torch.Tensor):
        x = self.preprocess(x)
        x = x.to(self.device)
        x = F.relu(x)

        with torch.set_grad_enabled(False):
            mesh = self.model.module.decoder.generate_mesh(x)
        return mesh


def reconstruct_all(recon, features, output_dir, source_layer, subject=None, roi=None):
    os.makedirs(os.path.join(output_dir, 'pointcloud'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mesh'), exist_ok=True)

    layer = source_layer
    if source_layer == 'encoder_bn5_relu':
        layer = 'encoder_bn5'
    elif source_layer == 'encoder_output_relu':
        layer = 'encoder_output'

    for label in tqdm(features.labels):
        print(label)

        try:
            if subject is not None:
                feat = features.get(layer=layer, subject=subject, roi=roi, label=label)
            else:
                feat = features.get(layer=layer, label=label)
        except Exception:
            print('Feature not found. Skipped.')
            continue

        pc_file = os.path.join(output_dir, 'pointcloud', label + '.npy')
        if not os.path.exists(pc_file):
            pc = recon(feat)
            np.save(pc_file, pc)
            print(f'Saved {pc_file}')

        mesh_file = os.path.join(output_dir, 'mesh', label + '.ply')
        if not os.path.exists(mesh_file):
            mesh = recon.generate_mesh(feat)
            mesh_processor.save(mesh, mesh_file, None)
            print(f'Saved {mesh_file}')


if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    output_root_dir = os.path.join(data_dir, 'reconstruction')

    models = [
        {
            'name':  'atlasnet',
            'path':  os.path.join(data_dir, 'models', 'atlasnet', 'network_crtd.pth'),
            'layer': 'encoder_bn5'
        },
    ]

    # True features
    true_datasets = [
        'test-3d-natural-objects',
    ]

    # Decoded features
    decoded_datasets = [
        'train-3d-natural-objects_rep3_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000_scaled_traincvstd',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000_scaled_traincvstd',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000_scaled_traincvstd',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000_scaled_traincvstd',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000_scaled_traincvstd',
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

        recon = AtlasNetAEReconFromFeature(
            device=CUDA,
            init_args={'model_path': model_path, 'source_layer': source_layer}
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

        recon = AtlasNetAEReconFromFeature(
            device=CUDA,
            init_args={'model_path': model_path, 'source_layer': source_layer}
        )

        reconstruct_all(recon, features, output_dir, source_layer, subject=sub, roi=roi)
