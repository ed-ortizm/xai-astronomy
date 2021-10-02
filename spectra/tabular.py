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

###############################################################################
ti = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("tabular.ini")
###############################################################################
# external imports
work_directory = parser.get("constants", "work")
ae_repository = parser.get("import", "ae")
anomaly_repository = parser.get("import", "anomaly")
sys.path.insert(0, f"{work_directory}")
sys.path.insert(0, f"{ae_repository}")
sys.path.insert(0, f"{anomaly_repository}")

from src.explainers.tabular import SpectraTabularExplainer
from variational.autoencoder import VAE
from reconstruction import ReconstructionAnomalyScore
###############################################################################
print(f"Creating explainer")
model_location = parser.get("directories", "model")
model = VAE.load(model_location)
mse = ReconstructionAnomalyScore(model).mse
regressor = partial(mse, percentage=10)
###############################################################################
train_data = np.load(parser.get("files", "train_data"))
explainer_parameters = dict(parser.items("explainer"))


explainer = SpectraTabularExplainer(train_data,
                                    explainer_parameters,
                                    regressor
                                    )
###############################################################################
spectrum = train_data[0]
reconstruction = spectrum + np.random.normal(size=(spectrum.shape))
explainer.explain_anomaly_score(
        spectrum,
        # number_features=,
)
###############################################################################
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
print(f"Running time: {tf-ti:.2f} s")
################################################################################
