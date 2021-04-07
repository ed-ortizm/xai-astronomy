#! /usr/bin/env python3

import glob
import time

import numpy as np

ti = time.time()
################################################################################
## Abreviations:
# exp(s): explanation(s)
################################################################################
data_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
output_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
################################################################################
## Loading relevant data
# The explanations
exp_files = glob.glob(f'{data_path}/outlier_feature_weight_k_size_*.npy')

print(f'Loading explanations for all kernels: [n_outlier, feature, weight]')

n_kernels = len(exp_files)
tmp = np.load(exp_files[0])
n_outliers = tmp.shape[0]
n_features = tmp.shape[1]
n_values = tmp.shape[2] + 1 # +1 to include the kernel value

tmp = np.empty((n_outliers, n_kernels, n_features, n_values))
# outlier_exps = {f'{int(n)}' : np.empty((n_kernels, n_features, n_values))
# for n in tmp[:, 0, 0]}

for idx_kernel, fname in enumerate(exp_files):

    n_kernel = float(fname.split('_')[-1].split('.')[0])
    tmp[:, idx_kernel, :, 1] = n_kernel

    exps = np.load(fname)

    for idx_outlier, exp in enumerate(exps):

        tmp[idx_outlier, idx_kernel, :, [0, 2, 3]] = exp[:, :].T

print(f'Saving final array')
np.save(f'{output_path}/exps_otl_ker_feat_weight.npy', tmp)

################################################################################
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
