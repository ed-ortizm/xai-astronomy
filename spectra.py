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
# if not os.path.exists(explanation_dir):
#     os.makedirs(explanation_dir)

# Loading training data
# train_set_name = f'{train_name}_spectra_{number_spectra}_{normalization_type}'
# train_set_path = f'{train_data_dir}/{train_set_name}.npy'
#
# train_data = load_data(train_set_name, train_set_path)
###############################################################################
# Loading training data
# test_set_name = f'spectra_{number_spectra}_{normalization_type}_nSnr_{number_spectra}_noSF_test'
# test_set_path = f'{train_data_dir}/{test_set_name}.npy'
# set_to_explain_path = f'{train_data_dir}/{set_to_explain_name}.npy'
#
# set_to_explain = load_data(set_to_explain_name, set_to_explain_path)
###############################################################################
# # Loading a reconstructed data
# tail_reconstructed = f'AE_{layers_str}_loss_{loss}'
# # spectra_500000_median_nSnr_500000_SF_train_reconstructed_AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
# reconstructed_set_name = (
#     f'{train_set_name}_reconstructed_{tail_reconstructed}_nTrain_{number_spectra}_nType_{normalization_type}')
#
# if local:
#     reconstructed_set_name = f'{reconstructed_set_name}_local'
#
# reconstructed_set_path = f'{generated_data_dir}/{reconstructed_set_name}.npy'
#
# reconstructed_data = load_data(reconstructed_set_name, reconstructed_set_path)
# ###############################################################################
# for metric in metrics:
#     # loading outlier scores of the training data
#     # mse_score_10_percent_train_AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
#     # AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
#     scores_name = (f'{set_to_explain_name}_{metric}_score_{percent_str}_'
#         f'{tail_outlier_name}')
#     # scores_name = f'{metric}_score_{percent_str}_train_{tail_outlier_name}'
#     # mse_score_10_percent
#     scores_name_path = (f'{generated_data_dir}/'
#         f'{set_to_explain_name}_{metric}_score_{percent_str}/'
#         f'{scores_name}.npy')
#     # scores_name_path = f'{generated_data_dir}/{metric}_score_{percent_str}/{scores_name}.npy'
#     scores = load_data(scores_name, scores_name_path)
#     ###############################################################################
    # loading top spectra

    # outlier_nTop_1000_mse_score_10_percent_test_AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
    # top_outlier_name = f'outlier_nTop_{number_top_spectra}_{metric}_score_{percent_str}_test_{tail_top_name}'
    # #top_normal_name = f'{metric}_normal_spectra_{percent_str}_{tail_top_name}'
    #
    # top_outlier_name_path = f'{generated_data_dir}/{metric}_score_{percent_str}/{top_outlier_name}.npy'
    # top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
    ################################################################################
    print(f"Creating explainers")
    # defining variables
    ################################################################################
    mode = parser.get('explainer', 'mode')
    kernel_width = np.sqrt(train_data[:, :-8].shape[1])*0.75
    # feature_selection: selects the features that have the highest
    # product of absolute weight * original data point when
    # learning with all the features
    feature_selection = parser.get('explainer', 'feature_selection')
    sample_around_instance = parser.get('explainer', 'sample_around')
    # feature_names = [i for i in range(train_data[:, :-8].shape[1])]
    ################################################################################
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
    # top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
    ################################################################################
    # outlier = Outlier(metric=metric, model=model)
    # outlier_score = partial(outlier.score, percentage=percent, image=False)
    ################################################################################
    # spectrum_explain = training_data[id_explain]
    # explanation_name_middle = f'{metric}_metric_{percent_str}'

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
