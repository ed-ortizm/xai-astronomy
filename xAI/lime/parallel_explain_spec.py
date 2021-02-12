#! /usr/bin/env python3
import glob
import os
import sys
import time

import csv
import numpy as np
from lib_explanations import Explainer, Explainer_parallel
from lib_explanations import Explanation
from lib_explanations import Outlier
# kk
ti = time.time()
################################################################################
# Naive parallelization
if len(sys.argv) > 1:
    simple = sys.argv[1] == "simple"
else:
    simple = False
################################################################################
# Relevant paths
training_data_file =\
    '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
training_data_path = '/home/edgar/zorro/SDSSdata/data_proc'
model_path = '/home/edgar/zorro/AEsII/trained_models/AutoEncoder'
o_scores_path = "/home/edgar/zorro/AEsII/outlier_scores"
spec_top_path = "/home/edgar/zorro/AEsII/xAI/lime/spec_top"
################################################################################
# Outlier scores to have a regression model
training_data = np.load(training_data_file)

# Training data files
if os.path.exists(f"./testing/training_data_files.tex"):
    with open('./testing/training_data_files.tex', 'r') as file:
        training_data_files = file.readlines()
else:
    training_data_files = glob.glob(f'{training_data_path}/*-*[0-9].npy')
    with open('./testing/training_data_files.tex', 'w') as file:
        file.writelines(f"{line}\n" for line in training_data_files)

feature_names = [f'flux {i}' for i in range(training_data.shape[1])]
kernel_width_default = np.sqrt(training_data.shape[1])*0.75

if simple:
# Creates a single explainer
    kernel_widths = [kernel_width_default]
    features_selection = ["highest_weights"]
    sample_around_instance = [False]

else:
# Creates many explainers
    kernel_widths = [kernel_width_default*weight for weight in
        np.hstack((np.linspace(0.1, 0.9, 9), np.linspace(1, 10, 10)))] #np.linspace(0.1, 1, 10)]
    features_selection = ["highest_weights", "lasso_path"] # , "none"]
    sample_around_instance = [True, False]


metrics = ["lp"]#, "mse", "chi2", "mad"]
pp = np.linspace(0.1, 1, 10)
n_spec = 20

for metric in metrics:

    if metric == "lp":
        p = 0.1
        outlier = Outlier(model_path = model_path, o_scores_path=o_scores_path,
            metric=metric, n_spec=n_spec, p=p)

    else:
        outlier = Outlier(model_path = model_path, o_scores_path=o_scores_path,
            metric=metric, n_spec=n_spec)

    print(f'Computing/loading outlier scores: the labels ')

    if os.path.exists(f'{o_scores_path}/{metric}_o_score.npy'):
        o_score_mse = np.load(f'{o_scores_path}/{metric}_o_score.npy')
    else:
        o_score_mse = outlier.score(O=training_data)
        np.save(f'{o_scores_path}/{metric}_o_score.npy', o_score_mse)

    t1 = time.time()
    print(f't1: {t1-ti:.2f} s')
    ################################################################################
    ## Selecting top outliers
    print(f'Computing top reconstructions for {metric} metric')
    most_normal, most_oulying = outlier.top_reconstructions(O=training_data)
    ###############################################################################
    print(f"Generating explanmations for the following outlying spectra")
    # From last array an by visual exploration, I'll like to explain:
    # tmp = [24, 28, 23, 21, 20, 19, 18, 17, 16]
    # tmp = [most_oulying[i] for i in tmp]
    # spec_2xpl = [training_data[i, :] for i in tmp]
    spec_2xpl = [training_data[i, :] for i in most_oulying]


    o_sdss_names = []
    o_sdss_paths = []

    for spec_idx in most_oulying:
        sdss_name, sdss_name_path = outlier.metadata(spec_idx=spec_idx,
        training_data_files=training_data_files)
        o_sdss_names.append(sdss_name)
        o_sdss_paths.append(sdss_name_path)
        np.save(f"{spec_top_path}/{sdss_name}.npy", np.load(sdss_name_path))

    # print(f"Working with the following outlying spectra")
    # for name in o_sdss_names:
    #     print(name)
    t2 = time.time()
    print(f't2: {t2-t1:.2f} s')
    ################################################################################
    ## Creating explainers in parallel

    # defining variables

    explainer_type="tabular"
    training_labels = o_score_mse

    print(f'Creating explainers')

    ti_exp = time.time()
    n_processes = 20
    tabular_explainers = Explainer_parallel(explainer_type, training_data,
        training_labels, feature_names, kernel_widths, features_selection,
        sample_around_instance)

    print(f'Generating explanations')

    for n, x in enumerate(spec_2xpl):
        explanations = tabular_explainers.get_explanations(
            x=x, regressor=outlier.score, sdss_name=o_sdss_names[n])
        tf_exp = time.time()
        # Saving explanations:
        if simple:
            with open(f"testing/{o_sdss_names[n]}_simple_{metric}.exp", 'w') as file:
                file.writelines(f"{line}\n" for line in explanations)
        else:
            with open(f"testing/{o_sdss_names[n]}_{metric}.exp", 'w') as file:
                file.writelines(f"{line}\n" for line in explanations)

    t3 = time.time()
    print(f't3: {t3-t2:.2f} s')
################################################################################
# The explanation wil be saved in a text file
# Generating explanations

# exp_list = tabular_explainer.explanation(x=spec_2xpl[0], regressor=outlier.score)
#
# with open(f'test_tmp.csv', 'w', newline='\n') as explanations_csv:
#     wr = csv.writer(explanations_csv, quoting=csv.QUOTE_ALL)
#     wr.writerow(exp_list)
#
# t4 = time.time()
# print(f't4: {t4-t3:.2f} s')
# ################################################################################
# # processing the explanation
# explanation = Explanation()
# wave_exp, flx_exp, weights_exp = explanation.analyze_explanation(spec_2xpl[0],
#     "test_tmp.csv")
#
# explanation.plot(spec_2xpl[0], wave_exp, flx_exp, weights_exp, show=True)
# # scatter lime weights vs mse outlier score
#
# ################################################################################
# ################################################################################
# ################################################################################
# # print(f"Generating explanmations for the most normal spectra")
# #
# # n_sdss_names = []
# # n_sdss_paths = []
# #
# # for spec_idx in most_normal:
# #     sdss_name, sdss_name_path = outlier.metadata(spec_idx,
# #     training_data_files=training_data_files)
# #     n_sdss_names.append(sdss_name)
# #     n_sdss_paths.append(sdss_name_path)
# #
# # print(n_sdss_names)
################################################################################
tf = time.time()
print(f'Running time for explanations: {tf_exp-ti_exp:.2f} s')
print(f'Running time: {tf-ti:.2f} s')