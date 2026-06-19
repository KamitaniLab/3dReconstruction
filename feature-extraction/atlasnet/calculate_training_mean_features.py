import os

from bdpy.dataform import Features, save_feature
import numpy as np


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

src_dir = os.path.join(data_dir, 'features', 'train-3d-natural-objects', 'atlasnet')
dst_dir = os.path.join(data_dir, 'features', 'train-3d-natural-objects', 'atlasnet_training_mean')

feat = Features(src_dir)

layers = feat.layers
for layer in layers:
    print(layer)
    f = feat.get(layer=layer)
    print(f.shape)
    f_mean = np.mean(f, axis=0, keepdims=True)
    print(f_mean.shape)

    save_feature(f_mean, dst_dir, layer=layer, label='mean', verbose=True)

print('DONE')
