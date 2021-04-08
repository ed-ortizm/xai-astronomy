#! /usr/bin/env python3
import glob
import os
import sys
import time

import lime
from lime import lime_tabular

import numpy as np

from libraryLime import workingDir, spectraDir

################################################################################
ti = time.time()
################################################################################
# Relevant paths
training_data_file =\
    '/home/edgar/zorro/SDSSdata/SDSS_data_curation/spec_99356.npy'
training_data_path = '/home/edgar/zorro/SDSSdata/data_proc'
model_path = '/home/edgar/zorro/AEs/trained_models/AutoEncoder'
o_scores_path = "/home/edgar/zorro/AEs/outlier_scores"
################################################################################
# Outlier scores to have a regression model

################################################################################
print(f"Creating explainer")
# defining variables
kernelWidthDefault = np.sqrt(trainingData.shape[1])*0.75
featureSelection = "highest_weights"
sampleAroundInstance = False
trainingLabels = o_score_mse
featureNames = [f"flux {i}" for i in range(training_data.shape[1])]

# explainer = Explainer(kernel_width=kernel_width_default,
#     feature_selection=feature_selection,
#     sample_around_instance=sample_around_instance,
#     explainer_type=explainer_type, training_data=training_data,
#     training_labels=training_labels, feature_names=feature_names)
#
# x = sys.getsizeof(explainer)*1e-6
# print(f'The size of the dilled explainer is: {x:.2f} Mbs')
#
# explanation = explainer.explanation(x=spec_2xpl[0], regressor=outlier.score)
#
# # Saving explanations:
# with open('testing/explain_spec.exp', 'w') as file:
#     file.writelines(f"{line}\n" for line in explanation)

# ################################################################################
# # processing the explanation
# explanation = Explanation()
# wave_exp, flx_exp, weights_exp = explanation.analyze_explanation(spec_2xpl[0],
#     "test_tmp.csv")
#
# explanation.plot(spec_2xpl[0], wave_exp, flx_exp, weights_exp, show=True)
# # scatter lime weights vs mse outlier score
#
# ################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
