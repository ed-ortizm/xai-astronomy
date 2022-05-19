"""Explain a single anomaly with LimeSpecExplainer"""
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from anomaly.reconstruction import ReconstructionAnomalyScore
from lime import lime_image
from astroExplain.spectra.segment import SpectraSegmentation
from autoencoders.ae import AutoEncoder
from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
# Set TensorFlow print of log information
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
config_file_name = "explainAnomaly.ini"
parser.read(f"{config_file_name}")
# Check files and directory
check = FileDirectory()
# Handle configuration file
configuration = ConfigurationFile()
###############################################################################
# set the number of cores to use with the reconstruction function
cores_per_worker = parser.getint("tensorflow-session", "cores")
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
print("Load anomalies", end="\n")

input_directory = parser.get("directory", "input")

anomalies_name = parser.get("file", "anomalies")
anomalies = np.load(f"{input_directory}/{anomalies_name}")

if anomalies.ndim == 2:
    # convert spectra to batch of gray images
    anomalies = anomalies[:, np.newaxis, :]
elif anomalies.ndim == 1:
    # convert single spectrum to gray image
    anomalies = anomalies[np.newaxis, np.newaxis, :]

wave_name = parser.get("file", "grid")
wave = np.load(f"{input_directory}/{wave_name}")
###############################################################################
# Load reconstruction function
print(f"Load reconstruction function", end="\n")

model_name = parser.get("file", "model")
model_directory = f"{input_directory}/{model_name}"
model = AutoEncoder(reload=True, reload_from=model_directory)
reconstruct_function = model.reconstruct

score_config = parser.items("score")
score_config = configuration.section_to_dictionary(score_config, [",", "\n"])
###############################################################################
# Load anomaly score function
print(f"Load anomaly score function", end="\n")
anomaly = ReconstructionAnomalyScore(
    reconstruct_function,
    wave,
    lines=score_config["lines"],
    velocity_filter=score_config["velocity"],
    percentage=score_config["percentage"],
    relative=score_config["relative"],
    epsilon=1e-3,
)
anomaly_score_function = partial(anomaly.score, metric=score_config["metric"])
###############################################################################
# Set explainer instance
print(f"Set explainer and Get explanations", end="\n")
explainer = lime_image.LimeImageExplainer(random_state=0)

number_segments = parser.getint("lime", "number_segments")
segmentation_fn = SpectraSegmentation().uniform
segmentation_fn = partial(segmentation_fn, number_segments=number_segments)

# Get explanations
save_explanation_to = parser.get("directory", "explanation")
save_explanation_to = f"{save_explanation_to}/{anomalies_name.split('.')[0]}"
check.check_directory(save_explanation_to, exit_program=False)

for idx, galaxy in enumerate(anomalies):

    print(f"Explain galaxy {idx}", end="\r")

    explanation = explainer.explain_instance(
        image=galaxy,
        classifier_fn=anomaly_score_function,
        labels=None,
        hide_color=parser.getint("lime", "hide_color"),
        top_labels=1,
        # num_features=1000, # default= 100000
        num_samples=parser.getint("lime", "number_samples"),
        batch_size=parser.getint("lime", "batch_size"),
        segmentation_fn=segmentation_fn
        # distance_metric="cosine",
    )

    save_name = f"{idx:05d}_explanationUniform"

    with open(f"{save_explanation_to}/{save_name}.pkl", "wb") as file:

        pickle.dump(explanation, file)


###############################################################################
with open(f"{save_explanation_to}/{config_file_name}", "w") as config_file:
    parser.write(config_file)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
