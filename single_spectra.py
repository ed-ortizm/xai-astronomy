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

parser.add_argument('--metric', type=str)
parser.add_argument('--top_spectra', '-top', type=int)
parser.add_argument('--percent', '-%', type=float)

parser.add_argument('--number_features', type=int)

parser.add_argument('--number_snr', '-n_snr', type=int)
parser.add_argument('--id_explain', '-id_xpl', type=str)


script_arguments = parser.parse_args()
################################################################################
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
local = script_arguments.server == 'local'

number_latent_dimensions = script_arguments.latent_dimensions
layers_encoder = script_arguments.encoder_layers
layers_decoder = script_arguments.decoder_layers

metric = script_arguments.metric
model = script_arguments.model
number_top_spectra = script_arguments.top_spectra

loss = script_arguments.loss

percent = script_arguments.percent
percent_str = f'{int(percent*100)}_percent'

id_explain = script_arguments.id_explain
number_features = script_arguments.number_features
number_snr = script_arguments.number_snr
################################################################################
# Relevant directories
################################################################################
if local:
    explanation_dir = f'{explanation_dir}_local'

if not os.path.exists(explanation_dir):
    os.makedirs(explanation_dir)
################################################################################
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'
training_data_dir = f'{spectra_dir}/processed_spectra'
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{number_spectra}'
###############################################################################
# Loading training data
train_set_name = f'spectra_{number_spectra}_{normalization_type}_nSnr_{number_snr}_SF_train'
train_set_path = f'{training_data_dir}/{train_set_name}.npy'

train_data = load_data(train_set_name, train_set_path)
###############################################################################
# Loading training data
test_set_name = f'spectra_{number_spectra}_{normalization_type}_nSnr_{number_snr}_noSF_test'
test_set_path = f'{training_data_dir}/{test_set_name}.npy'

#test_data = load_data(test_set_name, test_set_path)
###############################################################################
###############################################################################
# loading outlier scores
tail_outlier_name = f'{model}_{layers_str}_loss_{loss}_nTrain_{number_snr}_nType_{normalization_type}'

if local:
    tail_outlier_name = f'{tail_outlier_name}_local'
# mse_score_10_percent_train_AE_200_50_6_50_200_loss_mse_nTrain_500000_nType_median.npy
scores_name = f'{metric}_score_{percent_str}_train_{tail_outlier_name}'
# mse_score_10_percent
scores_name_path = f'{generated_data_dir}/{metric}_score_{percent_str}/{scores_name}.npy'
scores = load_data(scores_name, scores_name_path)
################################################################################
print(f"Creating explainers")
# defining variables
################################################################################
mode = 'regression'
kernel_width = np.sqrt(train_data[0, :-8].shape[0])*0.75
# feature_selection: selects the features that have the highest
# product of absolute weight * original data point when
# learning with all the features
feature_selection = 'highest_weights'
sample_around_instance = False
feature_names = [i for i in range(train_data[0, :-8].shape[0])]
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
# DenseDecoder_200_50_6_50_200_loss_mse_nTrain_20000_nType_median
model_head = f'{models_dir}/{model}/{layers_str}/Dense'
model_tail = f'{layers_str}_loss_{loss}_nTrain_{number_snr}_nType_{normalization_type}'
#    f'nType_{normalization_type}')
if local:
    model_tail = f'{model_tail}_local'

ae_path = f'{model_head}{model}_{model_tail}'
encoder_path = f'{model_head}Encoder_{model_tail}'
decoder_path = f'{model_head}Decoder_{model_tail}'

ae = LoadAE(ae_path, encoder_path, decoder_path)

percentages = [10., 20., 30., 40., 50., 75., 100.]

outlier = Outlier(metric=metric, model=ae)
outlier_score = partial(outlier.score, percentage=percent, image=False)
################################################################################
# spectrum_explain = training_data[id_explain]
explanation_name_middle = f'{metric}_metric_{percent}_percent'
explanation_name_tail = f'{model}_{model_tail}_explanation'

spectrum_explain = np.load(f'{id_explain}.npy')

explanation = explainer.explain_instance(
    data_row=spectrum_explain[:-8],
    predict_fn=outlier_score,
#        top_labels = 1,
    num_features=number_features)

spectrum_name = [f'{int(idx)}' for idx in spectrum_explain[-8:-5]]
spectrum_name = "_".join(spectrum_name)

if id_explain.split('-')[-1] == 'fake':
    spectrum_name = f'{spectrum_name}_fake'

explanation_name = (f'spectrum_{spectrum_name}_nFeatures_{number_features}_'
        f'{explanation_name_middle}_{explanation_name_tail}')

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
