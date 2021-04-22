#! /usr/bin/env python3
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
################################################################################
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

parser.add_argument('--id_explain', '-id_xpl', type=int)


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
percent_str = f'percentage_{int(percent*100)}'

id_explain = script_arguments.id_explain
################################################################################
# Relevant directories
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'
training_data_dir = f'{spectra_dir}/normalized_data'
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{number_spectra}'
###############################################################################
# Loading training data
train_set_name = f'spectra_{number_spectra}_{normalization_type}'
train_set_path = f'{training_data_dir}/{train_set_name}.npy'

training_data = load_data(train_set_name, train_set_path)
###############################################################################
# Loading a reconstructed data
tail_reconstructed = f'AE_{layers_str}_loss_{loss}'

reconstructed_set_name = (
    f'{train_set_name}_reconstructed_{tail_reconstructed}')

if local:
    reconstructed_set_name = f'{reconstructed_set_name}_local'

reconstructed_set_path = f'{generated_data_dir}/{reconstructed_set_name}.npy'

reconstructed_data = load_data(reconstructed_set_name, reconstructed_set_path)
###############################################################################
# loading outlier scores
tail_outlier_name = f'{model}_{layers_str}_loss_{loss}_{number_spectra}'

if local:
    tail_outlier_name = f'{tail_outlier_name}_local'

scores_name = f'{metric}_o_score_{percent_str}_{tail_outlier_name}'

scores_name_path = f'{generated_data_dir}/{scores_name}.npy'
scores = load_data(scores_name, scores_name_path)
################################################################################
print(f"Creating explainer")
# defining variables
################################################################################
# add model to predict a spec to explain because of the sampling than by lime
spectrum_explain = training_data[id_explain]
reconstructed_spectrum_explain = reconstructed_data[id_explain]

mode = 'regression'
kernel_width = np.sqrt(spectrum_explain[:-5].size)*0.75
# feature_selection: selects the features that have the highest
# product of absolute weight * original data point when
# learning with all the features
feature_selection = 'highest_weights'
sample_around_instance = False
feature_names = [i for i in range(spectrum_explain[:-5].size)]
################################################################################
explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data[:, :-5],
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
model_head = f'{models_dir}/{model}/{layers_str}/Dense'
model_tail = (f'{loss}_{layers_str}_nSpectra_{number_spectra}_'
    f'nType_{normalization_type}')
if local:
    model_tail = f'{model_tail}_local'

ae_path = f'{model_head}{model}_{model_tail}'
encoder_path = f'{model_head}Encoder_{model_tail}'
decoder_path = f'{model_head}Decoder_{model_tail}'

ae = LoadAE(ae_path, encoder_path, decoder_path)

outlier = Outlier(metric=metric, model=ae)
outlier_score = partial(outlier.score, percentage=percent)
# careful, I changed the outlier score.. check if top label keeps the results
################################################################################
explanation = explainer.explain_instance(
    data_row=spectrum_explain[:-5],
    predict_fn=outlier_score,
    top_labels = 1,
    num_features=100)

spectrum_name = [f'{int(idx)}' for idx in spectrum_explain[-5:-2]]
spectrum_name = "_".join(spectrum_name)

if not os.path.exists(explanation_dir):
    os.makedirs(explanation_dir)

with open(
    f'{explanation_dir}/spectrum_{spectrum_name}_fluxId_weight_explanation.txt',
    'w') as file:

    for explanation_weight in explanation.as_list():

        explanation_weight = (f'{explanation_weight[0]},'
            f'{explanation_weight[1]}\n')

        file.write(explanation_weight)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
