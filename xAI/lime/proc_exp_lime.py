#! /usr/bin/env python3

import time
import glob

import matplotlib.pyplot as plt
import numpy as np

ti = time.time()
################################################################################
output_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
lime_path ='/home/edgar/zorro/outlier_AEs/xAI/lime'

# Extracting explanation files paths and kernel widths
print(f'Extracting explanation files paths and kernel widths...')
# exp_files = glob.glob(f'{lime_path}/*_kernel/*.csv')
# k_widths = []
# for fname in exp_files:
#
#     width = fname.split('/')[-1].split('_')[0]
#     k_widths.append(int(width))

# Preparing explanations
#  test
fname = f'{lime_path}/test/outlier_nfeat_3801_exp_AE.csv'

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
print(n_outliers, n_features, n_values)
feature_weight = np.empty((n_outliers, n_features, n_values+1))
# n_values + 1 to add the outlier identifier

for outlier_idx, outlier_key in enumerate(explanations):

    exp = explanations[outlier_key]

    for feature_idx, tuple in enumerate(exp):

        tuple = tuple.split(',')
        # print(tuple)

        tuple[0] = np.float(tuple[0].strip("('flux")) - 1.0
        tuple[1] = np.float(tuple[1].strip(')'))

        tuple.append(outlier_idx)
        tuple[0], tuple[1], tuple[2] = tuple[2], tuple[0], tuple[1]
        feature_weight[outlier_idx, feature_idx, :] = np.array(tuple)

np.save(f'{lime_path}/test/outlier_feature_weight_nfeatt_{n_features}.npy',
feature_weight)
print(f'numpy array created: [n_outlier, feature, lime_weight]')

# for fname in exp_files:
#
#     k_size = int(fname.split('/')[-1].split('_')[0])
#     explanations = {}
#
#     with open(f'{fname}', newline='\n') as file:
#
#         for idx, line in enumerate(file):
#             explanations[f'{idx}_outlier'] = line
#
#     for outlier_key in explanations:
#
#         exp = explanations[outlier_key]
#         exp = exp.split('"')
#         exp = list(dict.fromkeys(exp))
#         exp.remove(',')
#         exp.remove('')
#         exp.remove('\r\n')
#
#         explanations[outlier_key] = exp
#
#
#     n_outliers = len(explanations)
#     n_features = len(exp)
#     n_values = len(exp[0].split(','))
#     # print(n_outliers, n_features, n_values)
#     feature_weight = np.empty((n_outliers, n_features, n_values+1))
#     # n_values + 1 to add the outlier identifier
#
#     for outlier_idx, outlier_key in enumerate(explanations):
#
#         exp = explanations[outlier_key]
#         for feature_idx, tuple in enumerate(exp):
#
#             tuple = tuple.split(',')
#
#             tuple[0] = tuple[0][2:-1]
#             tuple[1] = np.float(tuple[1][:-1])
#
#             if '<' in tuple[0]:
#
#                 if len(tuple[0].split('<'))==2:
#                     tuple[0] = np.int(tuple[0].split('<')[0])
#                 else:
#                     tuple[0] = np.int(tuple[0].split('<')[1])
#
#             else:
#                 # print(tuple[0].split('>')[0], tuple[0].split('>')[1])
#                 tuple[0] = np.int(tuple[0].split('>')[0])
#
#
#             tuple.append(outlier_idx)
#             tuple[0], tuple[1], tuple[2] = tuple[2], tuple[0], tuple[1]
#             feature_weight[outlier_idx, feature_idx, :] = np.array(tuple)
#
#     np.save(f'{output_path}/outlier_feature_weight_k_size_{k_size}.npy',
#     feature_weight)
# print(f'numpy array created: [n_outlier, feature, lime_weight]')

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
