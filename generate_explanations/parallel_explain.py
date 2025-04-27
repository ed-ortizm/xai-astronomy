"""Explain anomalies in parallel with LimeSpecExplainer"""
import argparse
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import os
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
import pandas as pd

import astroExplain.spectra.parallel as parallelExplainer
from astroExplain.spectra.utils import get_anomaly_score_name
from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile
from anomaly.constants import GALAXY_LINES

# Set environment variables to disable multithreading
# as users will probably want to set the number of cores
# to the max of their computer.    os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    """Get explanations in parallel"""

    parser = argparse.ArgumentParser(
        description="Parallel explanation of anomalies with LimeSpecExplainer"
        )

    parser.add_argument(
        "--config",
        type=str,
        default="parallel_explain.ini",
        help="Path to config file"
        )

    args = parser.parse_args()

    config_path = args.config

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_path)

    seed = parser.getint("configuration", "seed", fallback=0)
    np.random.seed(seed)
    #########################################################################
    start_time = time.perf_counter()

    config_handler = ConfigurationFile()

    mp.set_start_method("spawn", force=True)

    # Check files and directory
    check = FileDirectory()

    counter = mp.Value("i", 0)

    # strings to get right paths to data
    explanation_dir = parser.get("directory", "explanation")
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

    data_dir = parser.get("directory", "data")
    meta_data_dir = parser.get("directory", "meta")

    spectra_to_explain = np.load(
        f"{meta_data_dir}/{score_name}/{spectra_name}"
    )

    if spectra_to_explain.ndim == 1:
        spectra_to_explain = spectra_to_explain[np.newaxis, ...]

    spectra_to_explain_shape = spectra_to_explain.shape

    spectra_to_explain = RawArray(
        np.ctypeslib.as_ctypes_type(spectra_to_explain.dtype),
        spectra_to_explain.reshape(-1),
    )

    meta_data_spectra_name = parser.get("file", "meta")

    meta_data_spectra_df = pd.read_csv(
        f"{meta_data_dir}/{score_name}/{meta_data_spectra_name}",
        index_col="specobjid",
    )

    wave_name = parser.get("file", "grid")
    wave = np.load(f"{data_dir}/{wave_name}")
    wave = RawArray(np.ctypeslib.as_ctypes_type(wave.dtype), wave)

    model_dir = parser.get("directory", "model")

    check.check_directory(model_dir, exit_program=True)

    specobjid = np.array(
        meta_data_spectra_df.index,
        dtype=int
    )

    specobjid = RawArray(
        np.ctypeslib.as_ctypes_type(specobjid.dtype), specobjid.reshape(-1)
    )

    print("Load score and lime configurations")

    score_configuration = {}
    score_configuration["metric"] = metric
    score_configuration["lines"] = list(GALAXY_LINES.keys())
    score_configuration["velocity"] = parser.getfloat("score", "filter")

    if is_reconstruction is True:

        score_configuration["epsilon"] = parser.getfloat("score", "epsilon")
        score_configuration["relative"] = relative
        score_configuration["percentage"] = percentage


    lime_configuration = config_handler.section_to_dictionary(
        parser.items("lime"), [",", "\n"]
    )

    fudge_configuration = config_handler.section_to_dictionary(
        parser.items("fudge"), value_separators=[]
    )

    save_explanation_to = (
        f"{explanation_dir}/{score_name}"
    )

    check.check_directory(save_explanation_to, exit_program=False)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    explanation_str = parser.get("configuration", "explanation_str")
    save_explanation_to = (
        f"{save_explanation_to}/{timestamp}_{explanation_str}"
    )

    check.check_directory(f"{save_explanation_to}", exit_program=False)
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
            model_dir,
            save_explanation_to,
            cores_per_worker,
        ),
    ) as pool:

        pool.map(
            parallelExplainer.explain_anomalies,
            np.arange(spectra_to_explain_shape[0]),
        )

    with open(
        f"{save_explanation_to}/parallel_explain.ini", "w", encoding="utf8"
    ) as config_file:

        parser.write(config_file)
    finish_time = time.time()

    print(f"Run time: {finish_time - start_time:.2f}")

if __name__ == "__main__":
    main()
