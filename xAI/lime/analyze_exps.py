#! /usr/bin/env python3

import glob
import time

import matplotlib.pyplot as plt
import numpy as np

def plot_exp(n_outlier):

    """n_outlier: str"""

    # fig, ax

    pass

def spectra_outliers(data_path):
    """Return the spectra of outliers in dict"""

    spectra = {}
    # The outliers
    outlier_files = glob.glob(f'{data_path}/[0-9]_outlier.npy')

    for fname in outlier_files:

        n_outlier = fname.split('/')[-1].split('_')[0]
        spectra[n_outlier] = np.load(fname)

    return spectra

ti = time.time()
################################################################################
## Abreviations:
# exp(s): explanation(s)
################################################################################
data_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
img_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/images'
################################################################################
## Loading relevant data
exps = np.load(f'{data_path}/exps_otl_ker_feat_weight.npy')
spec_outliers = spectra_outliers(data_path)

for key_n_outlier in spec_outliers:

    spec = spec_outliers[key_n_outlier]

    wave_exps = exps[int(key_n_outlier), :, :, 2].astype(np.int)
    sort_idx_wave_exps = np.argsort(wave_exps, axis=1)

    flx_exps = np.empty((wave_exps.shape))

    for idx, sort_idx in enumerate(sort_idx_wave_exps):

        flx_exps[idx, :] = spec[wave_exps[idx, sort_idx]]

################################################################################
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
# # test
#
# outlier0 = np.load('0_outlier.npy')
# exp = np.load('outlier_feature_weight_nfeatt_3801.npy')
# wave_exp = exp[0, :, 1].astype(np.int)
# flx_exp = outlier0[wave_exp]
# weights = exp[0, :, 2]
# c = weights/np.max(weights)
# scatter(wave_exp, flx_exp, c=c, cmap='plasma_r')
# colorbar()
