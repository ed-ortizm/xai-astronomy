#! /usr/bin/env python3
from argparse import ArgumentParser
import glob
import os
import time

import lime
from lime import lime_tabular

import numpy as np

from constants_lime import working_dir, spectra_dir, normalization_schemes
from library_lime import load_data, Outlier
################################################################################
ti = time.time()
################################################################################
###############################################################################
parser = ArgumentParser()

parser.add_argument('--server', '-s', type=str)
parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--normalization_type', '-n_type', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--metric', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--top_spectra', '-top', type=int)
parser.add_argument('--loss', type=str)
parser.add_argument('--percent', '-%', type=str)

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
percent = f'percentage_{script_arguments.percent}'
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

reconstructed_set = load_data(reconstructed_set_name, reconstructed_set_path)
###############################################################################
# loading outlier scores
tail_outlier_name = f'{model}_{layers_str}_loss_{loss}_{number_spectra}'

if local:
    tail_outlier_name = f'{tail_outlier_name}_local'

scores_name = f'{metric}_o_score_{percent}_{tail_outlier_name}'

scores_name_path = f'{generated_data_dir}/{scores_name}.npy'
scores = load_data(scores_name, scores_name_path)
################################################################################
################################################################################
print(f"Creating explainer")
# defining variables
mode = 'regression'
kernel_width = np.sqrt(training_data.shape[1])*0.75
# feature_selection: selects the features that have the highest
# product of absolute weight * original data point when
# learning with all the features
feature_selection = 'highest_weights'
sample_around_instance = False
feature_names = [f"flux {i}" for i in range(training_data.shape[1])]
# ################################################################################
explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
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

outlier = Outlier(metric=metric)
# outlier_score = partial(outlier.score,
# # explanation = explainer.explain_instance(
# #     x=trainingData[0, :],
# #     regressor=,
# #     num_features=300)
# #         return xpl.as_list()
# # with open('testing/explain_spec.exp', 'w') as file:
# #     file.writelines(f"{line}\n" for line in explanation)

# ################################################################################
# ################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
