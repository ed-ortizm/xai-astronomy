#! /usr/bin/env python3
import glob
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

from lib_explanations import Explanation
######################################################################
ti = time.time()
################################################################################
lime_directory = "/home/edgar/zorro/AEsII/xAI/lime"
explanation_files = glob.glob(f'{lime_directory}/results/*.exp')
################################################################################
sdss_directory = "/home/edgar/zorro/SDSSdata/data_proc"
################################################################################
exp = Explanation()

fname = "spec-1246-54478-0144_exp_dict.dill"
with open(f'{lime_directory}/results/{fname}', 'rb') as file:
    explanation_dict = pickle.load(file)

for key, value in explanation_dict.items():
    k_width = float(value[0])
    explanation_array = value[1]
    metric = value[2]
    sdss_name = value[3]
    feature_selection = value[4]

    print(feature_selection, explanation_array.shape[0])

# # serialize explanations
# for explanation_file in explanation_files:
#
#     explanation_dict = exp.explanations_from_file(explanation_file)
#     sdss_name = explanation_file.split('/')[-1].split('_')[0]
#
#     with open(
#         f'{lime_directory}/results/{sdss_name}_exp_dict.dill', 'wb') as file:
#         pickle.dump(explanation_dict, file)
################################################################################
# wave_exp = explanation_array[:, 0].astype(np.int)
# flx_exp = spec[wave_exp]
# weights_exp = explanation_array[:, 1]
# print(spec.shape, wave_exp.shape)
#
# ################################################################################
# exp.plot(spec, wave_exp, flx_exp, weights_exp, linewidth=0.2,
#     cmap='plasma_r', show=True, ipython=True)
################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
