#! /usr/bin/env python3

import time

import csv
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model

ti = time.time()
################################################################################

# Loading model

model_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'
AE = load_model(f'{model_path}')

# Data used to train the model
train_data_path = '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
spec = np.load(f'{train_data_path}')

# Model predictions
pred = AE.predict(spec)

# Selecting top outliers for explanations
mse_score = np.square(pred-spec).mean(axis=1)
max_id = np.argmax(mse_score)
print(max_id)

# Creating explainers

list_file = open('explanations_list.csv', 'w', newline='\n')
k_widths = [5, 10, 20, 38, 50, 75, 100, 250, 500, 1_000]
for kernel_width in k_widths:

    print(f'Creating explainer...')
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=spec,
                mode='regression', training_labels = spec,
                kernel_width=kernel_width, verbose=True)

    # Generating an explanation
    print(f'Generating explanation...')
    exp = explainer.explain_instance(spec[max_id, :], AE.predict,
          num_features=100)

    print(f'Saving explanation as html')
    exp.save_to_file(file_path=f'./kw_{kernel_width}_explanation_AE.html')

    # explanation as list
    exp_list = exp.as_list()
    wr = csv.writer(list_file, quoting=csv.QUOTE_ALL)
    wr.writerow(exp_list)
    
    # explanation as pyplot figure
    exp_fig = exp.as_pyplot_figure()
    exp_fig.savefig(f'./kw_{kernel_width}_explanation_AE.png')

list_file.close()
# I must use the trained model to add a layer that computes the outlier
# score and that would be a new model which I want to explain
# outlier_exp = explainer.explain_instance(spec[0, :], np,
#       num_features=spec.shape[0])

################################################################################

tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
