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

# parser.add_argument('--id_explain', '-id_xpl', type=int)


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

# id_explain = script_arguments.id_explain
number_features = script_arguments.number_features
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
# loading top spectra
tail_top_name = (f'nTop_{number_top_spectra}_'
    f'{model}_{layers_str}_loss_{loss}_{number_spectra}')

# if local:
#     tail_outlier_name = f'{tail_outlier_name}_local'

top_outlier_name = f'{metric}_outlier_spectra_{percent_str}_{tail_top_name}'
top_normal_name = f'{metric}_normal_spectra_{percent_str}_{tail_top_name}'

top_outlier_name_path = f'{generated_data_dir}/{top_outlier_name}.npy'
top_outlier_spectra = load_data(top_outlier_name, top_outlier_name_path)
################################################################################
print(f"Creating explainers")
# defining variables
################################################################################
mode = 'regression'
kernel_width = np.sqrt(top_outlier_spectra[:, 1:-5].shape[1])*0.75
# feature_selection: selects the features that have the highest
# product of absolute weight * original data point when
# learning with all the features
feature_selection = 'highest_weights'
sample_around_instance = False
feature_names = [i for i in range(top_outlier_spectra[:, 1:-5].shape[1])]
################################################################################
model_head = f'{models_dir}/{model}/{layers_str}/Dense'

model_tail = (f'{loss}_{layers_str}_nSpectra_{number_spectra}_'
    f'nType_{normalization_type}')

if local:
    model_tail = f'{model_tail}_local'

# spectrum_explain = training_data[id_explain]

explanation_name_middle = f'{metric}_metric_{percent}_percent'
explanation_name_tail = f'{model}_{model_tail}_fluxId_weight_explanation'

for spectrum_explain in top_outlier_spectra:

    explanation = explainer.explain_instance(
        data_row=spectrum_explain[1:-5],
        predict_fn=outlier_score,
#        top_labels = 1,
        num_features=100)

    spectrum_name = [f'{int(idx)}' for idx in spectrum_explain[-5:-2]]
    spectrum_name = "_".join(spectrum_name)

    explanation_name = (f'spectrum_{spectrum_name}_nFeatures_{number_features}_'
        f'{explanation_name_middle}_{explanation_name_tail}')

    if local:
        explanation_name = f'{explanation_name}_local'
            file.write(explanation_weight)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
#
# ae_path = f'{model_head}{model}_{model_tail}'
# encoder_path = f'{model_head}Encoder_{model_tail}'
# decoder_path = f'{model_head}Decoder_{model_tail}'
#
# ae = LoadAE(ae_path, encoder_path, decoder_path)
#
# percentages = [10., 20., 30., 40., 50., 75., 100.]
#
# outlier = Outlier(metric=metric, model=ae)
# outlier_score = partial(outlier.score, percentage=percent, image=False)
