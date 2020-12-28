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

# The explanation
exp_files = glob.glob(f'{data_path}/outlier_feature_weight_k_size_*.npy')

# for fname in exp_files:

################################################################################
tf = time.time()

print(f'Running time: {tf-ti:.2f}')
