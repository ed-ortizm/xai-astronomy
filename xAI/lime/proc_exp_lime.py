#! /usr/bin/env python3

import time
import glob

import matplotlib.pyplot as plt
import numpy as np

ti = time.time()
################################################################################
# Extracting explanation files paths and kernel widths

print(f'Extracting explanation files paths and kernel widths...')
lime_path ='/home/edgar/zorro/outlier_AEs/xAI/lime'
exp_files = glob.glob(f'{lime_path}/*_kernel/*.csv')

k_widths = []
for fname in exp_files:

    width = fname.split('/')[-1].split('_')[0]
    k_widths.append(int(width))

# Preparing explanations
print(len(exp_files))
for fname in exp_files:

    explanations = {}

    with open(f'{fname}', newline='\n') as file:

        for idx, line in enumerate(file):
            explanations[f'{idx}_outlier'] = line

    for outlier_key in explanations:

        exp = explanations[outlier_key]
        exp = exp.split('"')
        exp = list(dict.fromkeys(exp))
        exp.remove(',')
        exp.remove('')
        exp.remove('\r\n')

        explanations[outlier_key] = exp


    n_outliers = len(explanations)
    n_features = len(exp)
    n_values = len(exp[0].split(','))
    # print(n_outliers, n_features, n_values)
    feature_weight = np.empty((n_outliers, n_features, n_values+1))
    # n_values + 1 to add the outlier identifier

    for outlier_idx, outlier_key in enumerate(explanations):

        exp = explanations[outlier_key]
        for feature_idx, tuple in enumerate(exp):

            tuple = tuple.split(',')

            tuple[0] = tuple[0][2:-1]
            tuple[1] = np.float(tuple[1][:-1])

            if '<' in tuple[0]:

                if len(tuple[0].split('<'))==2:
                    tuple[0] = np.int(tuple[0].split('<')[0])
                else:
                    tuple[0] = np.int(tuple[0].split('<')[1])

            else:
                # print(tuple[0].split('>')[0], tuple[0].split('>')[1])
                tuple[0] = np.int(tuple[0].split('>')[0])

            print(feature_idx, tuple)

            break
            tuple.append(outlier_idx)
            feature_weight[outlier_idx, feature_idx, :] = np.array(tuple)

    # print(feature_weight[1, 0, 1])
    # np.save('features_exp_weight.npy', feature_weight)

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
