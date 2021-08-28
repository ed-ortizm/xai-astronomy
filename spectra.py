#! /usr/bin/env python3
################################################################################
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import sys
import time
################################################################################
import lime
from lime import lime_tabular
############################################################################
import numpy as np
################################################################################
from library_outlier import Outlier
################################################################################
ti = time.time()
################################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('spectra.ini')
############################################################################
model = parser.get('models', 'model')

number_top_anomalies = parser.get('parameters', 'top_anomalies')
number_features = parser.get('parameters', 'features')
################################################################################
# number_spectra = script_arguments.number_spectra
# normalization_type = script_arguments.normalization_type
# local = script_arguments.server == 'local'
# metrics = script_arguments.metrics
# percent = script_arguments.percent
# train_name = script_arguments.train_name
# set_to_explain_name = script_arguments.explain_name
################################################################################
# Load model to explain
# model = LoadAE(ae_path, encoder_path, decoder_path)
################################################################################
# set_to_explain = load_data(set_to_explain_name, set_to_explain_path)
################################################################################
# Loading a reconstructed data
################################################################################
# loading top spectra
################################################################################
print(f"Creating explainers")
# defining variables
################################################################################
# kernel_width = np.sqrt(train_data[:, :-8].shape[1])*0.75
mode = parser.get('explainer', 'mode')
selection = highest_weights
sample = False
verbose = parser.getbool('explainer', 'verbose')
discretize_continuous = parser.getbool('explainer', 'discretize')
discretizer = parser.get('explainer', 'discretizer')
sample_around = parser.getbool('explainer', 'sample_around')
# stats = parser.getbool('explainer', 'stats')
feature_selection = parser.get('explainer', 'feature_selection')
sample_around_instance = parser.get('explainer', 'sample_around')
# feature_names = [i for i in range(train_data[:, :-8].shape[1])]
###############################################################################
# Gotta develop my class through inheritance
# explainer = lime_tabular.LimeTabularExplainer(
#             training_data=train_data[:, :-8],
#             mode=mode,
#             training_labels=scores,
#             feature_names=feature_names,
#             kernel_width=kernel_width,
#             verbose=True,
#             feature_selection=feature_selection,
#             discretize_continuous=False,
#             discretizer='quartile',
#             sample_around_instance=True,
#             training_data_stats=None)
################################################################################
# top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
################################################################################
 # outlier = Outlier(metric=metric, model=model)
 # outlier_score = partial(outlier.score, percentage=percent, image=False)
################################################################################
# spectrum_explain = training_data[id_explain]

# for spectrum_explain in top_outlier_spectra:

#      explanation = explainer.explain_instance(
#         data_row=spectrum_explain[1:-8],
#         predict_fn=outlier_score,
#         num_features=number_features)

#      with open(
#         f'{explanation_dir}/{explanation_name}.txt',
#         'w') as file:

#          for explanation_weight in explanation.as_list():

#              explanation_weight = (f'{explanation_weight[0]},'
#                 f'{explanation_weight[1]}\n')

#              file.write(explanation_weight)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
