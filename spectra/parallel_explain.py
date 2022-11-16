"""Explain anomalies in parallel with LimeSpecExplainer"""
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
import astroExplain.spectra.parallel as parallelExplainer
from astroExplain.spectra.utils import get_anomaly_score_name
from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

###############################################################################
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    ###########################################################################
    start_time = time.time()
    ###########################################################################
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_name = "parallel_explain.ini"
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
    explanation_directory = parser.get("directory", "explanation")
    spectra_name = parser.get("file", "spectra")

    metric = parser.get("score", "metric")
    velocity = parser.getint("score", "filter")

    is_reconstruction = len({"lp", "mad", "mse"}.intersection({metric})) != 0

    if is_reconstruction is True:

        relative = parser.getboolean("score", "relative")
        percentage = parser.getint("score", "percentage")

        score_name = get_anomaly_score_name(
            metric, velocity, relative, percentage
        )

    else:
        score_name = f"{metric}"

    spectra_to_explain = np.load(
        # f"{explanation_directory}/{score_name}/{spectra_name}"
        f"{explanation_directory}/{spectra_name}"
    )

    if spectra_to_explain.ndim == 1:
        spectra_to_explain = spectra_to_explain[np.newaxis, ...]

    spectra_to_explain_shape = spectra_to_explain.shape

    spectra_to_explain = RawArray(
        np.ctypeslib.as_ctypes_type(spectra_to_explain.dtype),
        spectra_to_explain.reshape(-1),
    )

    ###########################################################################
    meta_data_spectra_name = parser.get("file", "meta")
    meta_data_directory = parser.get("directory", "meta")

    meta_data_spectra_df = pd.read_csv(
        # f"{explanation_directory}/{score_name}/{meta_data_spectra_name}",
        f"{explanation_directory}/{meta_data_spectra_name}",
        index_col="specobjid",
    )
    ###########################################################################
    wave_name = parser.get("file", "grid")
    wave = np.load(f"{meta_data_directory}/{wave_name}")
    wave = RawArray(np.ctypeslib.as_ctypes_type(wave.dtype), wave)
    ###########################################################################
    model_id = parser.get("file", "model_id")
    model_directory = parser.get("directory", "model")
    model_directory = f"{model_directory}/{model_id}"
    check.check_directory(model_directory, exit_program=True)

    specobjid = np.array(meta_data_spectra_df.index, dtype=int)
    specobjid = RawArray(
        np.ctypeslib.as_ctypes_type(specobjid.dtype), specobjid.reshape(-1)
    )

    ###########################################################################
    print("Load score and lime configurations", end="\n")

    score_parser = ConfigParser(interpolation=ExtendedInterpolation())
    score_parser_name = parser.get("score", "configuration")
    score_parser.read(
        # f"{explanation_directory}/{score_name}/{score_parser_name}"
        f"{explanation_directory}/{score_parser_name}"
    )
    score_config = score_parser.items("score")
    score_config = configuration.section_to_dictionary(
        score_config, [",", "\n"]
    )

    score_configuration = {}
    score_configuration["metric"] = metric
    score_configuration["lines"] = score_config["lines"]
    score_configuration["velocity"] = velocity

    if is_reconstruction is True:

        score_configuration["epsilon"] = score_config["epsilon"]
        score_configuration["relative"] = relative
        score_configuration["percentage"] = percentage
    ###########################################################################

    lime_configuration = parser.items("lime")
    lime_configuration = configuration.section_to_dictionary(
        lime_configuration, [",", "\n"]
    )

    fudge_configuration = configuration.section_to_dictionary(
        parser.items("fudge"), value_separators=[]
    )
    ###########################################################################
    save_explanation_to = (
        f"{explanation_directory}/{score_name}/"
        # f"xai_{spectra_name.split('.')[0]}"
    )
    check.check_directory(save_explanation_to, exit_program=False)

    explanation_runs = glob.glob(f"{save_explanation_to}/*/")

    if len(explanation_runs) == 0:

        run = "000"

    else:

        runs = [int(run.split("/")[-2]) for run in explanation_runs]
        run = f"{max(runs)+1:03d}"

    save_explanation_to = f"{save_explanation_to}/{run}"
    check.check_directory(f"{save_explanation_to}", exit_program=False)
    ###########################################################################
    number_processes = parser.getint("configuration", "jobs")
    cores_per_worker = parser.getint("configuration", "cores_per_worker")

    with mp.Pool(
        processes=number_processes,
        initializer=parallelExplainer.init_shared_data,
        initargs=(
            counter,
            wave,
            specobjid,
            spectra_to_explain,
            spectra_to_explain_shape,
            score_configuration,
            lime_configuration,
            fudge_configuration,
            model_directory,
            save_explanation_to,
            cores_per_worker,
        ),
    ) as pool:

        pool.map(
            parallelExplainer.explain_anomalies,
            np.arange(spectra_to_explain_shape[0]),
        )

    ###########################################################################
    with open(
        f"{save_explanation_to}/{config_file_name}", "w", encoding="utf8"
    ) as config_file:

        parser.write(config_file)
    ###########################################################################
    finish_time = time.time()
    print(f"\nRun time: {finish_time - start_time:.2f}")
