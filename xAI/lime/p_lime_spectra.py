#! /usr/bin/env python3
import sys
import time

import csv
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model

from lib_lime import explain, mse_score, top_reconstructions

ti = time.time()
################################################################################
#
k_n = int(sys.argv[1])
# Data used to train the model
train_data_path = '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
outlier_score_path = '/home/edgar/zorro/outlier_AEs/xAI/lime/spectra_scores'
spec = np.load(f'{train_data_path}')

# # Selecting top outliers for explanations
o_score_mse = mse_score(O=spec)

# From visual exploration, check projec_gal_retireve for metadata of spectra
outliers_mse = np.load(f'{outlier_score_path}/outliers_mse.npy')
to_explain_spec_idx = [outliers_mse[i] for i in [24, 28, 23, 21, 20, 19, 18, 17, 16]]
outliers_to_exp = [spec[i, :] for i in to_explain_spec_idx]

k_widths = [5, 10, 20, 38, 50, 75, 100]


print(f'Creating explainer...')
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=spec,
            mode='regression', training_labels = o_score_mse,
            kernel_width=k_widths[k_n], verbose=True)

# Generating an explanation
num_features = spec.shape[1]
explanations_csv = open(
f'{k_widths[k_n]}_kernel/{k_widths[k_n]}_k_outlier_nfeat_{num_features}_exp_AE.csv', 'w', newline='\n')


for j, outlier in enumerate(outliers_to_exp):

    np.save(f'{j}_outlier.npy', outlier)
    # outlier = outlier.reshape(1, -1)

    print(f'Generating explanation...')
    exp = explainer.explain_instance(outlier, mse_score,
          num_features=num_features)

    print(f'Saving explanation as html')
    exp.save_to_file(
    file_path=\
    f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.html')


    # explanation as list
    exp_list = exp.as_list()
    wr = csv.writer(explanations_csv, quoting=csv.QUOTE_ALL)
    wr.writerow(exp_list)

    # explanation as pyplot figure
    exp_fig = exp.as_pyplot_figure()
    exp_fig.savefig(
    f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.png')

explanations_csv.close()
################################################################################

tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
