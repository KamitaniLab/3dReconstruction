import os
import copy
from glob import glob
from typing import List, Dict

from bdpy.dl.torch import DnnFeatureExtractorBase
from bdpy.dataform import save_feature
import numpy as np
import torch
from tqdm import tqdm

from models.autoencoder import AutoEncoder
from utils.misc import seed_all


# Global configuration #######################################################

RANDSEED = 42

# Feature extractor ##########################################################

class DiffusionAEFeatureExtractor(DnnFeatureExtractorBase):

    def init(self, model_path: str = '.', scale_factor_path: str = '.') -> None:
        ckpt = torch.load(model_path, map_location=self.device)
        seed_all(ckpt['args'].seed)
        self.model = self.model_cls(ckpt['args']).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        self.scale_factor = float(np.load(scale_factor_path))

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        '''
        Preprocess an input point cloud for DiffusionPC feature extraction.
        Rescales from AtlasNet scale (std ~ 0.29) to DiffusionPC training scale (std ~ 1).
        '''
        x = torch.Tensor(x) * self.scale_factor
        x = x.unsqueeze(0)  # Add batch dim: (1, N, 3)
        return x

    def extract_features(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        '''
        Extract features from DiffusionPC autoencoder encoder.
        'shape_latent' is the direct return value of encode(); other layers use forward hooks.
        '''
        def _store_feat(layer, module_in, module_out):
            _feat.append(module_out.cpu().numpy())

        x = x.to(self.device)
        features = {}

        with torch.no_grad():
            for layer in self.layers:
                model = copy.deepcopy(self.model)
                model = model.to(self.device)

                if layer == 'shape_latent':
                    features.update({'shape_latent': model.encode(x).cpu().numpy()})
                else:
                    exec(f'model.{layer}.register_forward_hook(_store_feat)')
                    _feat: List[np.ndarray] = []
                    model.encode(x)
                    features.update({layer: _feat[0]})

        return features


# Main #######################################################################

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    # Settings
    datasets = [
        {
           'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'train-3d-natural-objects', '*.npy')),
           'output_root_dir': os.path.join(data_dir, 'features', 'train-3d-natural-objects', 'diffusion_point_cloud')
        },
        {
           'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'test-3d-natural-objects', '*.npy')),
           'output_root_dir': os.path.join(data_dir, 'features', 'test-3d-natural-objects', 'diffusion_point_cloud')
        },
        {
            'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'test-3d-artificial-objects', '*.npy')),
            'output_root_dir': os.path.join(data_dir, 'features', 'test-3d-artificial-objects', 'diffusion_point_cloud')
        },
    ]

    model_path        = os.path.join(data_dir, 'models', 'diffusion_point_cloud', 'ckpt.pt')
    scale_factor_path = os.path.join(data_dir, 'models', 'diffusion_point_cloud', 'scaling_factor.npy')

    # Target layer setting
    encoder_layers = [
        # Intermediate encoder layers (PointNet-based encoder):
        'shape_latent',
    ]

    target_layers = encoder_layers

    # Initialize feature extractor
    feature_extractor = DiffusionAEFeatureExtractor(
        model_cls=AutoEncoder,
        layers=target_layers,
        device='cuda',
        init_args={
            'model_path':        model_path,
            'scale_factor_path': scale_factor_path,
        }
    )

    for dataset in datasets:
        pc_files = dataset['pc_files']
        output_root_dir = dataset['output_root_dir']

        if os.path.exists(output_root_dir):
            print(f'{output_root_dir} already exists. Skipped.')
            continue

        for pc_file in tqdm(pc_files):

            pc_name = os.path.basename(pc_file).split('.')[0]

            pc = np.load(pc_file)
            pc = torch.from_numpy(pc).float()

            # Downsampling to 2048 points (diffusion-point-cloud training size)
            if pc.size(0) > 2048:
                np.random.seed(RANDSEED)
                idx = np.random.choice(pc.size(0), 2048, replace=False)
                pc = pc[idx, :]

            # Feature extraction
            features = feature_extractor(pc)

            for layer, feature in features.items():
                layer_dirname = layer.replace('.', '_').replace('[', '_').replace(']', '')
                save_feature(feature, output_root_dir, layer=layer_dirname, label=pc_name, verbose=True)
