#! /usr/bin/env python3
import os
import sys
import time

import csv
import numpy as np
from lib_explain_spec import Explainer, Outlier, top_reconstructions

ti = time.time()
################################################################################
# Naive parallelization
if len(sys.argv) > 1:
    k_width = float(sys.argv[1])
else:
    k_width = None
################################################################################
# Relevant paths
training_data_path = '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
model_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'
################################################################################
# Outlier scores to have a regression model

# do here the if array exists so I don't have to compute it every time
print(f'Computing outlier scores: the labels ')

training_data = np.load(training_data_path)

if os.path.exists(f'{o_score_path}/outlier_score_mse.npy'):
    o_score_mse = np.load(f'{o_score_path}/outlier_score_mse.npy')
else:
    o_score_mse = Outlier(model_path=model_path).score(O=training_data,
    metric='mse')
    np.save(f'{o_score_path}/outlier_score_mse.npy', o_score_mse)

################################################################################
# Creating explainer

# ftr_selectt = ['highest_weights', 'lasso_path', 'none']
# training_data = np.load(training_data_path)
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

print(f'Creating explainer... feature selection: {ftr_select}')

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
