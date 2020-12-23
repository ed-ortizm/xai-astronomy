#! /usr/bin/env python3

import time

import csv
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model

from lib_lime import explain, mse_score, top_reconstructions

ti = time.time()
################################################################################

# # Loading trained model
# model_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'
# AE = load_model(f'{model_path}')

# Data used to train the model
train_data_path = '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
spec = np.load(f'{train_data_path}')
# pred = np.load('../../pred_99356.npy')
# # Selecting top outliers for explanations
o_score_mse = mse_score(O=spec)
#
# normal_mse, outliers_mse = top_reconstructions(scores=o_score_mse,
# n_normal_outliers=30)
#
# np.save(f'normal_mse.npy', normal_mse)
# np.save(f'outliers_mse.npy', outliers_mse)

# From visual exploration, check projec_gal_retireve for metadata of spectra
outliers_mse = np.load('outliers_mse.npy')
to_explain_spec_idx = [outliers_mse[i] for i in [24, 28, 23, 21, 20, 19, 18, 17, 16]]
outliers_to_exp = [spec[i, :] for i in to_explain_spec_idx]
# print(f'The id for the top MSE outlier is {max_id}')

k_widths = [5, 10, 20, 38, 50, 75, 100]

# if __name__ == "__main__":

# # Parallelization
#
#
#     n_cores = multiprocessing.cpu_count()
#
#     explanations_csv = open('explanations_list_p.csv', 'w', newline='\n')
#
#     # explainers_list = Parallel(n_jobs=n_cores)(delayed(explain)
#     # (kernel_width=kernel_width, training_data=spec, training_labels=spec,
#     # data_row = spec[max_id, :], predict_fn=AE.predict, num_features=100,
#     # file = explanations_csv) for kernel_width in k_widths)
#
#     with multiprocessing.Pool(processes=n_cores) as pool:
#         explainers_list = pool.starmap(explain,
#         ((kernel_width, spec, spec, spec[max_id, :], AE.predict, 100,
#         explanations_csv) for kernel_width in k_widths))
#
#     explanations_csv.close()

for kernel_width in k_widths:

    print(f'Creating explainer...')
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=spec,
                mode='regression', training_labels = o_score_mse,
                kernel_width=kernel_width, verbose=True)

    explanations_csv = open(
    f'{kernel_width}_kernel_outlier_explanations_list.csv', 'w', newline='\n')

    # Generating an explanation
    for j, outlier in enumerate(outliers_to_exp):

        np.save(f'{j}_outlier.npy', outlier)
        # outlier = outlier.reshape(1, -1)

        print(f'Generating explanation...')
        exp = explainer.explain_instance(outlier, mse_score,
              num_features=100)

        print(f'Saving explanation as html')
        exp.save_to_file(
        file_path=f'./{j}_outlier_kw_{kernel_width}_explanation_AE.html')


        # explanation as list
        exp_list = exp.as_list()
        wr = csv.writer(explanations_csv, quoting=csv.QUOTE_ALL)
        wr.writerow(exp_list)

        # explanation as pyplot figure
        exp_fig = exp.as_pyplot_figure()
        exp_fig.savefig(f'./{j}_outlier_kw_{kernel_width}_explanation_AE.png')

    explanations_csv.close()


# I must use the trained model to add a layer that computes the outlier
# score and that would be a new model which I want to explain
# outlier_exp = explainer.explain_instance(spec[0, :], np,
#       num_features=spec.shape[0])

################################################################################

tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
