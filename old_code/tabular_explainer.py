#! /usr/bin/env python3
import sys
from argparse import ArgumentParser
from functools import partial
import os
import time

import lime
from lime import lime_tabular

import numpy as np

from constants_lime import normalization_schemes
from constants_lime import explanation_dir, models_dir, spectra_dir, working_dir
from library_lime import load_data, LoadAE
from library_outlier import Outlier
################################################################################
ti = time.time()
###############################################################################
parser = ArgumentParser()

parser.add_argument('--server', '-s', type=str)

parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--normalization_type', '-n_type', type=str)

parser.add_argument('--model', type=str)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--loss', type=str)

parser.add_argument('--metrics', type=str, nargs='+')
parser.add_argument('--top_spectra', '-top', type=int)
parser.add_argument('--percent', '-%', type=int)

parser.add_argument('--number_features', type=int)

#parser.add_argument('--number_spectra', '-n_snr', type=int)

parser.add_argument('--train_name', type=str)
parser.add_argument('--explain_name', type=str)


script_arguments = parser.parse_args()
################################################################################
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
local = script_arguments.server == 'local'

layers_encoder = script_arguments.encoder_layers
number_latent_dimensions = script_arguments.latent_dimensions
layers_decoder = script_arguments.decoder_layers
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'

metrics = script_arguments.metrics
model = script_arguments.model
number_top_spectra = script_arguments.top_spectra

loss = script_arguments.loss

percent = script_arguments.percent
percent_str = f'{percent}_percent'

# id_explain = script_arguments.id_explain
number_features = script_arguments.number_features

train_name = script_arguments.train_name
set_to_explain_name = script_arguments.explain_name

################################################################################
# Loading auto encoder
model_head = f'{models_dir}/{model}/{layers_str}/Dense'

model_tail = (f'{layers_str}_loss_{loss}_nTrain_{number_spectra}_'
    f'nType_{normalization_type}')

if local:
    model_tail = f'{model_tail}_local'

ae_path = f'{model_head}{model}_{model_tail}'
encoder_path = f'{model_head}Encoder_{model_tail}'
decoder_path = f'{model_head}Decoder_{model_tail}'

ae = LoadAE(ae_path, encoder_path, decoder_path)
################################################################################
# Relevant directories
################################################################################
if local:
    explanation_dir = f'{explanation_dir}_local'

if not os.path.exists(explanation_dir):
    os.makedirs(explanation_dir)
################################################################################
train_data_dir = f'{spectra_dir}/processed_spectra'
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{number_spectra}'
###############################################################################
# Loading training data
train_set_name = f'{train_name}_spectra_{number_spectra}_{normalization_type}'
train_set_path = f'{train_data_dir}/{train_set_name}.npy'

train_data = load_data(train_set_name, train_set_path)
###############################################################################
# Loading training data
# test_set_name = f'spectra_{number_spectra}_{normalization_type}_nSnr_{number_spectra}_noSF_test'
# test_set_path = f'{train_data_dir}/{test_set_name}.npy'
set_to_explain_path = f'{train_data_dir}/{set_to_explain_name}.npy'

set_to_explain = load_data(set_to_explain_name, set_to_explain_path)
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
tail_outlier_name = f'{model}_{model_tail}'
    # # (f'{model}_{layers_str}_loss_{loss}_nTrain_{number_spectra}_'
    # #     f'nType_{normalization_type}')

for metric in metrics:
    # loading outlier scores of the training data
    # mse_score_10_percent_train_AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
    # AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
    scores_name = (f'{set_to_explain_name}_{metric}_score_{percent_str}_'
        f'{tail_outlier_name}')
    # scores_name = f'{metric}_score_{percent_str}_train_{tail_outlier_name}'
    # mse_score_10_percent
    scores_name_path = (f'{generated_data_dir}/'
        f'{set_to_explain_name}_{metric}_score_{percent_str}/'
        f'{scores_name}.npy')
    # scores_name_path = f'{generated_data_dir}/{metric}_score_{percent_str}/{scores_name}.npy'
    scores = load_data(scores_name, scores_name_path)
    ###############################################################################
    # loading top spectra
    tail_top_name = tail_outlier_name

    #(f'nTop_{number_top_spectra}_'
    #    f'{model}_{layers_str}_loss_{loss}_{number_spectra}')

    # if local:
    #     tail_outlier_name = f'{tail_outlier_name}_local'
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
    mode = 'regression'
    kernel_width = np.sqrt(train_data[:, :-8].shape[1])*0.75
    # feature_selection: selects the features that have the highest
    # product of absolute weight * original data point when
    # learning with all the features
    feature_selection = 'highest_weights'
    sample_around_instance = False
    feature_names = [i for i in range(train_data[:, :-8].shape[1])]
    ################################################################################
    explainer = lime_tabular.LimeTabularExplainer(
                training_data=train_data[:, :-8],
                mode=mode,
                training_labels=scores,
                feature_names=feature_names,
                kernel_width=kernel_width,
                verbose=True,
                feature_selection=feature_selection,
                discretize_continuous=False,
                discretizer='quartile',
                sample_around_instance=True,
                training_data_stats=None)
    ################################################################################
    top_outlier_name = f'outlier_nTop_{number_top_spectra}_{scores_name}'
    # top_outlier_name = f'outlier_nTop_{number_top_spectra}_{metric}_score_{percent_str}_test_{tail_top_name}'
    #top_normal_name = f'{metric}_normal_spectra_{percent_str}_{tail_top_name}'

    top_outlier_name_path = (f'{generated_data_dir}/'
        f'{set_to_explain_name}_{metric}_score_{percent_str}/{top_outlier_name}.npy')

    top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
    # top_outlier_name_path = f'{generated_data_dir}/{metric}_score_{percent_str}/{top_outlier_name}.npy'
    # top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
    ################################################################################
    outlier = Outlier(metric=metric, model=ae)
    outlier_score = partial(outlier.score, percentage=percent, image=False)
    ################################################################################
    # spectrum_explain = training_data[id_explain]
    # explanation_name_middle = f'{metric}_metric_{percent_str}'
    explanation_name_tail = f'{scores_name}_explanation'

    for spectrum_explain in top_outlier_spectra:

        explanation = explainer.explain_instance(
            data_row=spectrum_explain[1:-8],
            predict_fn=outlier_score,
            num_features=number_features)

        spectrum_name = [f'{int(idx)}' for idx in spectrum_explain[-8:-5]]
        spectrum_name = "-".join(spectrum_name)

        explanation_name = (f'spec-{spectrum_name}_nFeatures_{number_features}_'
            f'{explanation_name_tail}')

        if local:
            explanation_name = f'{explanation_name}_local'

        with open(
            f'{explanation_dir}/{explanation_name}.txt',
            'w') as file:

            for explanation_weight in explanation.as_list():

                explanation_weight = (f'{explanation_weight[0]},'
                    f'{explanation_weight[1]}\n')

                file.write(explanation_weight)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
