#! /usr/bin/env python3
import glob
import os
import sys
import time

import csv
import numpy as np
from lib_explain_spec import Explainer, Outlier
# kk
ti = time.time()
################################################################################
# Naive parallelization
if len(sys.argv) > 1:
    k_width = float(sys.argv[1])
else:
    k_width = None
################################################################################
# Relevant paths
training_data_file =\
    '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
training_data_path = '/home/edgar/zorro/SDSSdata/data_proc'
model_path = '/home/edgar/zorro/AEs/trained_models/AutoEncoder'
o_scores_path = "/home/edgar/zorro/AEs/outlier_scores"
################################################################################
# Outlier scores to have a regression model
training_data = np.load(training_data_file)

metric = 'mse'
n_spec = 30
outlier = Outlier(model_path = model_path, o_scores_path=o_scores_path,
    metric=metric, n_spec=n_spec)

print(f'Computing/loading outlier scores: the labels ')
print(o_scores_path)
if os.path.exists(f'{o_scores_path}/{metric}_o_score.npy'):
    o_score_mse = np.load(f'{o_scores_path}/{metric}_o_score.npy')
else:
    o_score_mse = outlier.score(O=training_data)
    np.save(f'{o_scores_path}/{metric}_o_score.npy', o_score_mse)

################################################################################
## Selecting top outliers
print(f'Computing top reconstructions for {metric} metric')
most_normal, most_oulying = outlier.top_reconstructions(O=training_data)
################################################################################
print(f"Generating explanmations for the following outlying spectra")
# From last array an by visual exploration, I'll like to explain:
tmp = [24, 28, 23, 21, 20, 19, 18, 17, 16]
tmp = [most_oulying[i] for i in tmp]
spec_2xpl = [training_data[i, :] for i in tmp]

# Training data files
training_data_files = glob.glob(f'{training_data_path}/*-*[0-9].npy')

print(f"Generating explanmations for the following outlying spectra")

o_sdss_names = []
o_sdss_paths = []

for spec_idx in most_oulying:
    sdss_name, sdss_name_path = outlier.metadata(spec_idx,
    training_data_files=training_data_files)
    o_sdss_names.append(sdss_name)
    o_sdss_paths.append(sdss_name_path)

print(o_sdss_names)
################################################################################

# Creating explainer

## defining variables

# ftr_selectt = ['highest_weights', 'lasso_path', 'none']
# training_data = np.load(training_data_file)
training_labels = o_score_mse
feature_names = [f'flux {i}' for i in range(training_data.shape[1])]
kernel_width = k_width
feature_selection = 'highest_weights'
training_data_stats = None
sample_around_instance = False
discretize_continuous = False
discretizer = 'decile'
verbose = True
mode = 'regression'

# print(f'Creating explainer... feature selection: {feature_selection}')

################################################################################
# print(f"Generating explanmations for the most normal spectra")
#
# n_sdss_names = []
# n_sdss_paths = []
#
# for spec_idx in most_normal:
#     sdss_name, sdss_name_path = outlier.metadata(spec_idx,
#     training_data_files=training_data_files)
#     n_sdss_names.append(sdss_name)
#     n_sdss_paths.append(sdss_name_path)
#
# print(n_sdss_names)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
