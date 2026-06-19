import os
import copy
from glob import glob
from typing import List, Dict

from bdpy.dl.torch import DnnFeatureExtractorBase
from bdpy.dataform import save_feature
import numpy as np
import torch
from torchvision import transforms
import PIL
from PIL import Image
from tqdm import tqdm
import json

from model.model import EncoderDecoder
from easydict import EasyDict
import dataset.pointcloud_processor as pointcloud_processor


# Global configuration #######################################################

RANDSEED = 42

# Feature extractor ##########################################################

class AtlasNetAutoEncoderFeatureExtractor(DnnFeatureExtractorBase):

    def init(self, model_path: str = '.', model_opt: str = '.') -> None:

        with open(model_opt, 'r') as f:
            model_opts = json.load(f)
        model_opts = EasyDict(model_opts)
        model_opts['device'] = torch.device('cuda')

        self.model = self.model_cls(model_opts)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        '''
        Preprocess an input point cloud for AtlasNet feature extraction.
        '''

        x = torch.Tensor(x)

        x = x.unsqueeze(0)  # Add batch dim

        x = x.transpose(2, 1).contiguous()

        return x

    def extract_features(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        '''
        Extract AtlasNet features from the input image.
        '''

        def _store_feat(layer, module_in, module_out):
            _feat.append(module_out.cpu().numpy())

        x = x.to(self.device)

        features = {}

        with torch.set_grad_enabled(False):

            for layer in self.layers:
                model = copy.deepcopy(self.model)
                model = model.to(self.device)

                exec(f'model.{layer}.register_forward_hook(_store_feat)')

                _feat: List[np.ndarray] = []
                model(x, train=False)

                features.update({layer: _feat[0]})

        return features


# Main #######################################################################

if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    # Settings
    datasets = [
        {
           'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'train-3d-natural-objects', '*.npy')),
           'output_root_dir': os.path.join(data_dir, 'features', 'train-3d-natural-objects', 'atlasnet')
        },
        {
           'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'test-3d-natural-objects', '*.npy')),
           'output_root_dir': os.path.join(data_dir, 'features', 'test-3d-natural-objects', 'atlasnet')
        },
        {
            'pc_files': glob(os.path.join(data_dir, 'pointcloud', 'test-3d-artificial-objects', '*.npy')),
            'output_root_dir': os.path.join(data_dir, 'features', 'test-3d-artificial-objects', 'atlasnet')
        },
    ]

    model_path     = os.path.join(data_dir, 'models', 'atlasnet', 'network_crtd.pth')
    model_opt_path = os.path.join(data_dir, 'models', 'atlasnet', 'options.json')

    # Target layer setting
    encoder_layers = [
        # 'encoder.conv1',
        # 'encoder.conv2',
        # 'encoder.conv3',
        # 'encoder.lin1',
        # 'encoder.lin2',
        'encoder.bn5' # In the paper, we used only this layer.
    ]

    # decoder_block = [
    #     'decoder.decoder[{}].conv2',
    #     'decoder.decoder[{}].conv_list[0]',
    #     'decoder.decoder[{}].conv_list[1]',
    #     'decoder.decoder[{}].last_conv'
    # ]

    # patch_num = 1

    target_layers = encoder_layers
    # for i in range(patch_num):
    #     target_layers.extend([d.format(i) for d in decoder_block])

    # Initialize feature extractor
    feature_extractor = AtlasNetAutoEncoderFeatureExtractor(
        model_cls=EncoderDecoder,
        layers=target_layers,
        device='cuda',
        init_args={
            'model_path': model_path,
            'model_opt':  model_opt_path,
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

            # Downsampling
            if pc.size(0) > 2500:
                np.random.seed(RANDSEED)
                idx = np.random.choice(pc.size(0), 2500, replace=True)
                pc = pc[idx, :]

            # Feature extraction
            features = feature_extractor(pc)

            for layer, feature in features.items():
                layer_dirname = layer.replace('.', '_').replace('[', '_').replace(']', '')  # Is this necessary?
                save_feature(feature, output_root_dir, layer=layer_dirname, label=pc_name, verbose=True)
