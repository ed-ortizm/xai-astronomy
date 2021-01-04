#! /usr/bin/env python3
import os
import sys
import time

import csv
import numpy as np
from lib_lime import mse_score, top_reconstructions

ti = time.time()
################################################################################
if len(sys.argv) > 1:
    k_width = float(sys.argv[1])
else:
    k_width = None
## Relevant paths
train_data_path = '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
o_score_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
model_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'

## Data used to train the model

print('Loading relevant data')
spec = np.load(f'{train_data_path}')

# do here the if array exists so I don't have to compute it every time
print(f'Computing outlier scores: the labels ')

if os.path.exists(f'{o_score_path}/outlier_score_mse.npy'):
    o_score_mse = np.load(f'{o_score_path}/outlier_score_mse.npy')
else:
    o_score_mse = mse_score(O=spec, model_path = model_path)
    np.save(f'{o_score_path}/outlier_score_mse.npy', o_score_mse)

## Selecting top outliers for explanations
# check projec_gal_retireve for metadata of spectra

print(f'Loading top outliers')
# This can be a function
n_normal_outliers = 30
if os.path.exists(
                f'{o_score_path}/top_{n_normal_outliers}_outliers_idx_mse.npy'):
    top_oo_mse = np.load(
                f'{o_score_path}/top_{n_normal_outliers}_outliers_idx_mse.npy')
else:
    top_normal_mse, top_oo_mse = top_reconstructions(scores=o_score_mse,
                                        n_normal_outliers=n_normal_outliers)
    np.save(f'{o_score_path}/top_{n_normal_outliers}_normal_idx_mse.npy',
            top_normal_mse)
    np.save(f'{o_score_path}/top_{n_normal_outliers}_outliers_idx_mse.npy',
            top_oo_mse)

# From last array an by visual exploration, I'll like to explain:
tmp = [24, 28, 23, 21, 20, 19, 18, 17, 16]

tmp = [top_oo_mse[i] for i in tmp]
spec_2xpl = [spec[i, :] for i in tmp]
################################################################################
t1 = time.time()
time1 = t1-ti
print(f'Running time 1: {time1:.2f} s')
################################################################################
## Explanations
ftrr_names = [f'flux {i+1}' for i in range(spec.shape[1])]
#'highest_weights': selects the features that have the highest product of
# absolute weight * original data point when learning with all the features
ftr_selectt = ['highest_weights', 'lasso_path', 'none', 'forward_selection']
# spec_stats = data_stats(data=spec)

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
