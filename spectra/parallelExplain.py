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
import glob
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
import pandas as pd
import tensorflow as tf

import astroExplain.spectra.parallel as parallelExplainer
from autoencoders.ae import AutoEncoder
from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    ###########################################################################
    start_time = time.time()
    ###########################################################################
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_name = "parallelExplain.ini"
    parser.read(f"{config_file_name}")
    ###########################################################################
    # Check files and directory
    check = FileDirectory()
    # Handle configuration file
    configuration = ConfigurationFile()
    ###########################################################################
    counter = mp.Value("i", 0)
    ###########################################################################
    # strings to get right paths to data
    data_directory = parser.get("directory", "data")
    data_bin = parser.get("common", "bin")

    # Load data
    print("Load anomalies")

    anomalies_file = parser.get("file", "anomalies")
    anomalies_name = anomalies_file.split(".")[0]

    # if testing
    is_test = parser.getboolean("configuration", "test")

    if is_test is True:

        score_directory = f"{data_directory}/{data_bin}/test"

    else:

        score_directory = f"{data_directory}/{data_bin}/{anomalies_name}"

    check.check_directory(score_directory, exit=True)

    anomalies = np.load(f"{score_directory}/{anomalies_file}")

    specobjid = anomalies[:, 0].astype(int)
    share_specobjid = RawArray(
        np.ctypeslib.as_ctypes_type(specobjid.dtype), specobjid.reshape(-1)
    )
    del specobjid

    # load spectra

    fluxes = np.load(f"{data_directory}/fluxes.npy", mmap_mode="r")
    anomalies_indexes = anomalies[:, 1].astype(int)
    anomalies = fluxes[anomalies_indexes]

    del fluxes

    if anomalies.ndim == 1:
        anomalies = anomalies[np.newaxis, ...]

    share_anomalies = RawArray(
        np.ctypeslib.as_ctypes_type(anomalies.dtype), anomalies.reshape(-1)
    )

    anomalies_shape = anomalies.shape
    del anomalies

    ###########################################################################
    wave_name = parser.get("file", "grid")
    wave = np.load(f"{data_directory}/{wave_name}")
    share_wave = RawArray(np.ctypeslib.as_ctypes_type(wave.dtype), wave)

    del wave

    ###########################################################################
    print(f"Load score and lime configurations", end="\n")
    # Fetch lines and epsilon from configuration file used to get scores
    anomalies_parser = ConfigParser(interpolation=ExtendedInterpolation())
    anomalies_configuration = parser.get("file", "configuration_anomalies")
    anomalies_parser.read(f"{score_directory}/{anomalies_configuration}")

    ###########################################################################
    score_configuration = {}

    lines = anomalies_parser.get("score", "lines")
    lines = configuration.entry_to_list(lines, str, "\n")
    score_configuration["lines"] = lines

    epsilon = anomalies_parser.getfloat("score", "epsilon")
    score_configuration["epsilon"] = epsilon

    ##########################################################################
    temp = anomalies_name.split(".")[0].split("_")

    if len(temp) == 2:
        # lp_rel100.npy

        metric = temp[0]
        velocity = 0

        if "no" in temp[1]:
            relative = False
            percentage = float(temp[1].strip("noRel"))

        else:
            relative = True
            percentage = float(temp[1].strip("rel"))

    else:
        # lp_filter_50Kms_rel100.npy

        metric = temp[0]
        velocity = float(temp[2].strip("Kms"))

        if "no" in temp[3]:
            relative = False
            percentage = float(temp[3].strip("noRel"))

        else:
            relative = True
            percentage = float(temp[3].strip("rel"))

    score_configuration["metric"] = metric
    score_configuration["velocity"] = velocity
    score_configuration["relative"] = relative
    score_configuration["percentage"] = percentage

    ###########################################################################

    lime_configuration = parser.items("lime")
    lime_configuration = configuration.section_to_dictionary(
        lime_configuration, [",", "\n"]
    )
    ###########################################################################
    model_directory = f"{data_directory}/{data_bin}/models"
    model_name = parser.get("file", "model")
    model_directory = f"{model_directory}/{model_name}"
    check.check_directory(model_directory, exit=True)

    save_explanation_to = f"{score_directory}/explanations"

    explanation_runs = glob.glob(f"{save_explanation_to}/*/")

    if len(explanation_runs) == 0:

        run = "00000"

    else:

        runs = [int(run.split("/")[-2]) for run in explanation_runs]
        run = f"{max(runs)+1:05d}"

    save_explanation_to = f"{save_explanation_to}/{run}"
    check.check_directory(f"{save_explanation_to}", exit=False)
    ###########################################################################
    number_processes = parser.getint("configuration", "jobs")
    cores_per_worker = parser.getint("configuration", "cores_per_worker")

    with mp.Pool(
        processes=number_processes,
        initializer=parallelExplainer.init_shared_data,
        initargs=(
            counter,
            share_wave,
            share_specobjid,
            share_anomalies,
            anomalies_shape,
            score_configuration,
            lime_configuration,
            model_directory,
            save_explanation_to,
            cores_per_worker,
        ),
    ) as pool:

        pool.map(
            parallelExplainer.explain_anomalies, np.arange(anomalies_shape[0])
        )

    ###########################################################################
    with open(f"{save_explanation_to}/{config_file_name}", "w") as config_file:
        parser.write(config_file)
    ###########################################################################
    finish_time = time.time()
    print(f"\nRun time: {finish_time - start_time:.2f}")
