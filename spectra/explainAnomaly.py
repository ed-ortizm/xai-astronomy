import os

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
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import time
import pickle

from lime import lime_image
import numpy as np
import tensorflow as tf

from astroExplain.segmentation import SpectraSegmentation
from astroExplain.toyRegressors import SpectraPlus
from autoencoders.ae import AutoEncoder
from sdss.superclasses import ConfigurationFile, FileDirectory

###############################################################################
start_time = time.time()
###############################################################################
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("explainAnomaly.ini")
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
print("Load anomalies" , end="\n")
input_directory = parser.get("directory", "input")

anomalies_name = parser.get("file", "anomalies")
anomalies = np.load(f"{input_directory}/{anomalies_name}")
meta_data_directory = parser.get("directory", "meta_data")
wave_name = parser.get("file", "grid")
wave = np.load(f"{meta_data_directory}/{wave_name}")
###############################################################################
# Load reconstruction function
print(f"Load reconstruction function", end="\n")

model_directory = parser.get("directory", "model")
model = AutoEncoder(reload=True, reload_from=model_directory)
reconstruct_function = model.reconstruct
#
# score_config = parser.items("score")
# score_config = configuration.section_to_dictionary(score_config, [",", "\n"])
#
# save_to = parser.get("directory", "output")
# check.check_directory(save_to, exit=False)
# ###############################################################################
# # Set explainer instance
# explainer = lime_image.LimeImageExplainer(random_state=0)
#
# number_segments = parser.getint("lime", "number_segments")
# segmentation_fn = SpectraSegmentation().uniform
# segmentation_fn = partial(segmentation_fn, number_segments=number_segments)
# # get explanation
#
# explanation = explainer.explain_instance(
#     image=galaxy[np.newaxis, ...], # image.dim == 2
#     classifier_fn=addSpectra.predict,
#     labels=None,
#     hide_color=1, # the spectrum is median normalized
#     top_labels=1,
#     # num_features=1000, # default= 100000
#     num_samples=1_000,
#     batch_size=10,
#     segmentation_fn=segmentation_fn
#     # distance_metric="cosine",
# )
#
# print(f"Finish explanation... Saving...", end="\n")
#
# save_name = f"{name_galaxy}ExplanationUniform"
#
# with open(f"{output_directory}/{save_name}.pkl", "wb") as file:
#
#     pickle.dump(explanation, file)
# ###############################################################################
# finish_time = time.time()
# print(f"Run time: {finish_time-start_time:.2f}")
