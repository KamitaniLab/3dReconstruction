from itertools import product
import os
import pickle

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pairwise_identification
import numpy as np
import pandas as pd
import yaml

# List of feature decoding configs
FEATURE_DECODING_CONFIG_FILES = [
    "config/train-3d-natural-objects-image_rep3_fmap_test-3d-natural-objects-image_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml",
    "config/train-3d-natural-objects-image_rep3_fmap_test-3d-artificial-objects-image_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml",
    "config/train-3d-natural-objects-image_rep3_fmap_test-3d-artificial-objects-rds_rep8_fmap_fmriprep_5000voxel_atlasnet.yaml",
]


# List of lures for pair-wise identification analysis
LURES_DIRS = [
    "../data/features/test-3d-natural-objects/atlasnet",
    "../data/features/test-3d-artificial-objects/atlasnet",
]


def pairwise_identification_analysis(config_file: str, lures: Features, output_file: str):
    '''Pair-wise identification analysis.'''
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    decoded_feature_path = cfg["decoded_feature"]["path"]
    true_feature_path = cfg["decoded_feature"]["features"]["paths"][0]
    subjects = [s["name"] for s in cfg["decoded_feature"]["fmri"]["subjects"]]
    rois = [r["name"] for r in cfg["decoded_feature"]["fmri"]["rois"]]
    layers = cfg["decoded_feature"]["features"]["layers"]

    print(f'Decoded features: {decoded_feature_path}')
    print(f'True features:    {true_feature_path}')
    print(f'Subjects: {subjects}')
    print(f'ROIs:     {rois}')
    print(f'Layers:   {layers}')

    features_test = Features(true_feature_path)
    decoded_features = DecodedFeatures(decoded_feature_path)

    if os.path.exists(output_file):
        print(f'Loading {output_file}')
        with open(output_file, 'rb') as f:
            results = pickle.load(f)
    else:
        print('Creating new results store')
        results = pd.DataFrame(columns=['layer', 'subject', 'roi', 'profile_correlation', 'identification_accuracy'])

    for layer in layers:
        print(f'Layer: {layer}')

        true_y = features_test.get(layer=layer)
        true_labels = features_test.labels

        # Lure features: drop entries whose label overlaps with the true set
        lure_y_all = lures.get(layer=layer)
        lure_labels = lures.labels
        true_label_set = set(true_labels)
        keep = [i for i, l in enumerate(lure_labels) if l not in true_label_set]
        lure_y = lure_y_all[keep]
        print(f'  Lures: {len(keep)} (excluded {len(lure_labels) - len(keep)} overlapping)')

        for subject, roi in product(subjects, rois):
            print(f'  Subject: {subject} - ROI: {roi}')

            done = ((results['layer'] == layer) & (results['subject'] == subject) & (results['roi'] == roi)).any()
            if done:
                print('    Already done. Skipped.')
                continue

            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)
            pred_labels = decoded_features.selected_label
            pred_labels = [l.replace('_light-sun-front-left', '').replace('rds_', '') for l in pred_labels]

            pred_y_selected = []
            pred_labels_selected = []
            for i, l in enumerate(pred_labels):
                if l not in true_labels:
                    continue
                pred_y_selected.append(pred_y[i])
                pred_labels_selected.append(l)
            pred_y = np.stack(pred_y_selected, axis=0)
            pred_labels = pred_labels_selected

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            candidates = np.concatenate([true_y_sorted, lure_y], axis=0)

            r_prof = profile_correlation(pred_y, true_y_sorted)
            print(f'    Mean profile correlation:     {np.nanmean(r_prof)}')

            ident = pairwise_identification(pred_y, candidates)
            print(f'    Mean identification accuracy: {np.nanmean(ident)}')

            results = pd.concat([results, pd.DataFrame([{
                'layer': layer,
                'subject': subject,
                'roi': roi,
                'profile_correlation': r_prof,
                'identification_accuracy': ident,
            }])], ignore_index=True)

            with open(output_file, 'wb') as f:
                pickle.dump(results, f)

    print('All done')
    return output_file


if __name__ == "__main__":
    # Load lures
    print(f"Loading lures from {LURES_DIRS}")
    lures = Features(LURES_DIRS)
    print(f"  Layers: {lures.layers}")
    print(f"  Number of lures: {len(lures.labels)}")

    # Run pair-wise identification analysis
    for config_file in FEATURE_DECODING_CONFIG_FILES:
        pairwise_identification_analysis(config_file, lures=lures, output_file="eval.pkl")
