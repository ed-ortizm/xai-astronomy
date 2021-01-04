#! /usr/bin/env python3
import os
import sys
import time

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
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
# k_widths = [5, 10, 20, 38, 50, 75, 100]
#
ftrr_names = [f'flux {i+1}' for i in range(spec.shape[1])]
#'highest_weights': selects the features that have the highest product of
# absolute weight * original data point when learning with all the features
ftr_selectt = ['highest_weights', 'lasso_path', 'none', 'forward_selection']
# spec_stats = data_stats(data=spec)
for ftr_select in ftr_selectt:
    print(f'Creating explainer... feature selection: {ftr_select}')
    discretize_continuous = False
    discretizer = 'decile'
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=spec,
                mode='regression', training_labels=o_score_mse,
                feature_names=ftrr_names, kernel_width=k_width, verbose=True,
                feature_selection=ftr_select,
                discretize_continuous=discretize_continuous, discretizer= discretizer,
                sample_around_instance=False, training_data_stats=None)
    # Kernel ($\Pi(x)$) and kernel width ($\sigma$ [?]) are the default parameters
    ################################################################################
    t2 = time.time()
    time2 = t2-t1
    print(f'Running time 2: {time2:.2f} s')
    ################################################################################

    # Generating an explanation
    # test
    # number of features to include in the explanation. This K <--> $\Omega$
    num_features = spec.shape[1]
    explanations_csv = open(
    f'test/nfeat_{num_features}_discretize_{int(discretize_continuous)}_{discretizer}_ftr_select_{ftr_select}.csv',
    'w', newline='\n')

    # num_features = spec.shape[1]

    # explanations_csv = open(
    # f'{k_widths[k_n]}_kernel/{k_widths[k_n]}_k_outlier_nfeat_{num_features}_exp_AE.csv', 'w', newline='\n')


    for j, outlier in enumerate(spec_2xpl):
    # test
        np.save(f'test/{j}_outlier.npy', outlier)
        # np.save(f'{o_score_path}/{j}_outlier.npy', outlier)

        print(f'Generating explanation...')
        exp = explainer.explain_instance(outlier, mse_score,
              num_features=num_features)

        print(f'Saving explanation as html')
    # test
        exp.save_to_file(
        file_path = f'test/nfeat_{num_features}_discretize_{int(discretize_continuous)}_{discretizer}_ftr_select_{ftr_select}.html')
        # exp.save_to_file(
        # file_path=\
        # f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.html')


        # explanation as list
        exp_list = exp.as_list()
        wr = csv.writer(explanations_csv, quoting=csv.QUOTE_ALL)
        wr.writerow(exp_list)

        # explanation as pyplot figure
    # test
        exp_fig = exp.as_pyplot_figure()
        exp_fig.savefig(f'test/nfeat_{num_features}_discretize_{int(discretize_continuous)}_{discretizer}_ftr_select_{ftr_select}.png')

        # exp_fig = exp.as_pyplot_figure()
        # exp_fig.savefig(
        # f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.png')

        break

    explanations_csv.close()
################################################################################
t3 = time.time()
time3 = t3-t2
print(f'Running time 3: {time3:.2f} s')
################################################################################

################################################################################
# Writing code in a neat way
## Abbreviations

# ftr: feature
# o: outlier
# oo: outliers
# pv: previous variable
# idx: index
# idxx: indexes
# xpl: explain
# Kernel width will be introduced from the command newline
# k_widths = []


# print(f'Creating explainer...')
# explainer = lime.lime_tabular.LimeTabularExplainer(training_data=spec,
#             mode='regression', training_labels = o_score_mse,
#             kernel_width=k_widths[k_n], verbose=True)
#
# # Generating an explanation
# num_features = spec.shape[1]
# explanations_csv = open(
# f'{k_widths[k_n]}_kernel/{k_widths[k_n]}_k_outlier_nfeat_{num_features}_exp_AE.csv', 'w', newline='\n')
#
#
# for j, outlier in enumerate(outliers_to_exp):
#
#     np.save(f'{j}_outlier.npy', outlier)
#     # outlier = outlier.reshape(1, -1)
#
#     print(f'Generating explanation...')
#     exp = explainer.explain_instance(outlier, mse_score,
#           num_features=num_features)
#
#     print(f'Saving explanation as html')
#     exp.save_to_file(
#     file_path=\
#     f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.html')
#
#
#     # explanation as list
#     exp_list = exp.as_list()
#     wr = csv.writer(explanations_csv, quoting=csv.QUOTE_ALL)
#     wr.writerow(exp_list)
#
#     # explanation as pyplot figure
#     exp_fig = exp.as_pyplot_figure()
#     exp_fig.savefig(
#     f'{k_widths[k_n]}_kernel/{j}_outlier_k_{k_widths[k_n]}_nfeat_{num_features}_exp_AE.png')
#
# explanations_csv.close()

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
