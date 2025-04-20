"""Explain a single anomaly with LimeSpecExplainer"""
import os

# disable tensorflow logs: warnings and info :). Allow error logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#########################################################################
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import pickle
import time

import tensorflow as tf
import numpy as np

from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile
from autoencoders.ae import AutoEncoder
from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from astroExplain.spectra.explainer import LimeSpectraExplainer
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.utils import get_anomaly_score_name

#########################################################################
start_time = time.time()
#########################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "explain.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
#########################################################################
# set the number of cores to use with the reconstruction function
cores_per_worker = parser.getint("tensorflow", "cores")
jobs = cores_per_worker
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=jobs,
    inter_op_parallelism_threads=jobs,
    allow_soft_placement=True,
    device_count={"CPU": jobs},
)
session = tf.compat.v1.Session(config=config)
###############################################################################
# Load data
print("Load spectra to explain", end="\n")

explanation_directory = parser.get("directory", "explanation")

spectra_name = parser.get("file", "spectra")

metric = parser.get("score", "metric")
velocity = parser.getint("score", "filter")
relative = parser.getboolean("score", "relative")
percentage = parser.getint("score", "percentage")

score_name = get_anomaly_score_name(metric, velocity, relative, percentage)

spectra_to_explain = np.load(
    f"{explanation_directory}/{score_name}/{spectra_name}"
)

if spectra_to_explain.ndim == 2:
    # convert spectra to batch of gray images
    spectra_to_explain = spectra_to_explain[:, np.newaxis, :]
elif spectra_to_explain.ndim == 1:
    # convert single spectrum to gray image
    spectra_to_explain = spectra_to_explain[np.newaxis, np.newaxis, :]

#########################################################################
meta_data_directory = parser.get("directory", "meta")
wave_name = parser.get("file", "grid")
wave = np.load(f"{meta_data_directory}/{wave_name}")
#########################################################################
# Load reconstruction function
print("Load reconstruction function", end="\n")

model_id = parser.get("file", "model_id")
model_directory = parser.get("directory", "model")
model = AutoEncoder(reload=True, reload_from=f"{model_directory}/{model_id}")
reconstruct_function = model.reconstruct
#########################################################################
# Load anomaly score function
score_parser = ConfigParser(interpolation=ExtendedInterpolation())
score_parser_name = parser.get("score", "configuration")
score_parser.read(f"{explanation_directory}/{score_name}/{score_parser_name}")
score_config = score_parser.items("score")
score_config = configuration.section_to_dictionary(score_config, [",", "\n"])


filter_parameters = FilterParameters(
    wave=wave, velocity_filter=velocity, lines=score_config["lines"]
)

reconstruction_parameters = ReconstructionParameters(
    relative=relative, percentage=percentage, epsilon=1e-3
)

print("Load anomaly score function", end="\n")
anomaly = ReconstructionAnomalyScore(
    reconstruct_function,
    reconstruction_parameters=reconstruction_parameters,
    filter_parameters=filter_parameters,
)
anomaly_score_function = partial(anomaly.score, metric=metric)
#########################################################################
# Set explainer instance
print("Set explainer and Get explanations", end="\n")
explainer = LimeSpectraExplainer(random_state=0)

number_segments = parser.getint("lime", "number_segments")
segmentation_fn = SpectraSegmentation().uniform
segmentation_fn = partial(segmentation_fn, number_segments=number_segments)

# Get explanations

save_explanation_to = (
    f"{explanation_directory}/{score_name}/"
    f"xai_{spectra_name.split('.')[0]}"
)
check.check_directory(save_explanation_to, exit_program=False)

fudge_parameters = configuration.section_to_dictionary(
    parser.items("fudge"), value_separators=[]
)
explainer_parameters = configuration.section_to_dictionary(
    parser.items("lime"), value_separators=[]
)

for idx, galaxy in enumerate(spectra_to_explain):

    print(f"Explain galaxy {idx}", end="\r")

    explanation = explainer.explain_instance(
        spectrum=galaxy,
        classifier_fn=anomaly_score_function,
        segmentation_fn=segmentation_fn,
        fudge_parameters=fudge_parameters,
        explainer_parameters=explainer_parameters,
    )

    save_name = f"{idx:05d}_explanationUniform"

    with open(f"{save_explanation_to}/{save_name}.pkl", "wb") as file:

        pickle.dump(explanation, file)


#########################################################################
# with open(f"{save_explanation_to}/{config_file_name}", "w") as config_file:
#     parser.write(config_file)
#########################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
