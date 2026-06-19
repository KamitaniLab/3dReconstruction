import os
from itertools import product

from bdpy.dataform import Features, DecodedFeatures, save_array
import numpy as np


def feature_scaling(dec_feat, train_feat, train_feat_cv, output_dir):
    '''
    Scale decoded features and save them.

    Parameters
    ----------
    dec_feat : str
      Path to decoded features.
    train_feat : str
      Path to true features used for decoder training.
    output_dir : str
      Path to save scaled decoded features.

    Returns
    -------
    None
    '''
    print('----------------------------------------')
    print(f'Decoded features:  {dec_feat}')
    print(f'Training features: {train_feat}')
    print(f'Training CV:       {train_feat_cv}')
    print(f'Output:            {output_dir}')

    tf = Features(train_feat)
    df = DecodedFeatures(dec_feat)
    cv = DecodedFeatures(train_feat_cv)

    layers = df.layers

    for layer in layers:
        f_train_mean = tf.statistic(layer=layer, statistic='mean').squeeze()
        f_train_std  = tf.statistic(layer=layer, statistic='std').squeeze()

        subjects = df.subjects
        rois = df.rois
        labels = df.labels

        for subject, roi in product(subjects, rois):

            if layer not in df.layers or subject not in df.subjects or roi not in df.rois:
                continue

            if layer not in cv.layers or subject not in cv.subjects or roi not in cv.rois:
                continue

            try:
                f_cv = cv.get(layer=layer, subject=subject, roi=roi)
            except RuntimeError:
                print(f"Failed to load CV features for {layer}, {subject}, and {roi}. Skipped.")
                continue

            for label in labels:
                output_file = os.path.join(
                    output_dir,
                    layer, subject, roi,
                    f'{label}.mat'
                )
                if os.path.exists(output_file):
                    continue

                try:
                    f = df.get(layer=layer, subject=subject, roi=roi, label=label)
                except RuntimeError:
                    print(f"Failed to load decoded features for {layer}, {subject}, {roi}, and {label}. Skipped.")
                    continue

                f_scaled = (f.squeeze() - f_train_mean) * (f_train_std / np.std(f_cv, axis=0).squeeze()) + f_train_mean
                f_scaled = f_scaled[np.newaxis]

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                save_array(output_file, f_scaled, key='feat', dtype=np.float32, sparse=False)
                print(f'Saved {output_file}')

    return None


if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

    network = 'atlasnet'

    train_feature_dataset = 'train-3d-natural-objects'

    train_features_cv = os.path.join(
        data_dir, 'decoded-features',
        'train-3d-natural-objects_rep3_fmap_fmriprep_5000voxel_allunits_fastl2lir_alpha5000',
        network,
    )

    decoded_experiments = [
        'train-3d-natural-objects_rep3_test-3d-natural-objects_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-horizontal-shape-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
        'train-3d-natural-objects_rep3_test-3d-contour-matched-rds-thin-tilt-variants_rep8_fmap_fmriprep_5000voxel_fastl2lir_alpha5000',
    ]

    train_feat_path = os.path.join(data_dir, 'features', train_feature_dataset, network)

    for exp in decoded_experiments:
        dec_feat_path = os.path.join(data_dir, 'decoded-features', exp, network)
        output_path   = os.path.join(data_dir, 'decoded-features', exp + '_scaled_traincvstd', network)

        feature_scaling(dec_feat_path, train_feat_path, train_features_cv, output_path)
