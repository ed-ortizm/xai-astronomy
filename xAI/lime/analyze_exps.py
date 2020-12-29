#! /usr/bin/env python3

import glob
import time

import numpy as np

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


################################################################################
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
