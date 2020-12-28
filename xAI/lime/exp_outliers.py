#! /usr/bin/env python3

import glob
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ti = time.time()
################################################################################
## Abreviations:
# exp(s): explanation(s)
################################################################################
data_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
output_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
################################################################################
# Loading relevant data

# The outliers
outlier_files = glob.glob(f'{data_path}/[0-9]_outlier.npy')
m = len(outlier_files)
n = np.load(outlier_files[0]).size
spectra = np.empty((m,n))

for idx, fname in enumerate(outlier_files):

    spectra[idx, :] = np.load(fname)

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
    # print(n_kernel)
    for idx_outlier, exp in enumerate(exps):

        # exp = np.hstack(exp[],np.array([n_kernel]), )
        tmp[idx_outlier, idx_kernel, :, [0, 2, 3]] = exp[:, :].T
        # tmp[idx_outlier, idx_kernel, :, 2:] = exp[:, 1:]

        break
    for row in tmp[0, idx_kernel, 0, :]:
        print(row)
    break

################################################################################
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
