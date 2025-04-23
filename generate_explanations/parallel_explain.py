"""Explain anomalies in parallel with LimeSpecExplainer"""
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import glob
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

def main():
    """Get explanations in parallel"""
    # Set environment variables to disable multithreading
    # as users will probably want to set the number of cores
    # to the max of their computer.    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(
        description="Train a VAE using config file."
        )

    parser.add_argument(
        "--config",
        type=str,
        default="train.ini",
        help="Path to config file"
        )
    args = parser.parse_args()

    config_path = args.config
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_path)
    seed = parser.getint("hyperparaneters", "seed", fallback=0)
    np.random.seed(seed)
    #########################################################################
    start_time = time.perf_counter()
    ########################################################################
    config_handler = ConfigurationFile()
    ########################################################################

    mp.set_start_method("spawn", force=True)

    # Check files and directory
    check = FileDirectory()

    counter = mp.Value("i", 0)

    # strings to get right paths to data
    explanation_dir = parser.get("directory", "explanation")
    spectra_name = parser.get("file", "spectra")

    print(f"explanation_dir: {explanation_dir}")
    print(f"spectra_name: {spectra_name}")

    metric = parser.get("score", "metric")
    velocity = parser.getint("score", "filter")
    print(f"metric: {metric}")
    print(f"velocity: {velocity}")

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
    print(f"score_name: {score_name}")
    print(f"spec to explain: {data_dir}/{score_name}/{spectra_name}")

    spectra_to_explain = np.load(
        f"{data_dir}/{score_name}/{spectra_name}"
    )

    if spectra_to_explain.ndim == 1:
        spectra_to_explain = spectra_to_explain[np.newaxis, ...]

    spectra_to_explain_shape = spectra_to_explain.shape
    print(f"spectra_to_explain_shape: {spectra_to_explain_shape}")

    spectra_to_explain = RawArray(
        np.ctypeslib.as_ctypes_type(spectra_to_explain.dtype),
        spectra_to_explain.reshape(-1),
    )

    meta_data_spectra_name = parser.get("file", "meta")
    meta_data_dir = parser.get("directory", "meta")
    print(f"meta_data_dir: {meta_data_dir}")
    print(f"meta_data_spectra_name: {meta_data_spectra_name}")
    meta_data_spectra_df = pd.read_csv(
        f"{data_dir}/{score_name}/{meta_data_spectra_name}",
        # f"{explanation_dir}/{meta_data_spectra_name}",
        index_col="specobjid",
    )

    wave_name = parser.get("file", "grid")
    wave = np.load(f"{meta_data_dir}/{wave_name}")
    wave = RawArray(np.ctypeslib.as_ctypes_type(wave.dtype), wave)

    model_name = parser.get("file", "model")
    model_dir = parser.get("directory", "model")
    model_dir = f"{model_dir}/{model_name}"
    print(f"model_dir: {model_dir}")
    # check.check_dir(model_dir, exit_program=True)

    specobjid = np.array(meta_data_spectra_df.index[:100], dtype=int)
    specobjid = RawArray(
        np.ctypeslib.as_ctypes_type(specobjid.dtype), specobjid.reshape(-1)
    )

    print("Load score and lime configurations", end="\n")

    score_parser = ConfigParser(interpolation=ExtendedInterpolation())
    score_parser_name = parser.get("score", "configuration")
    print(f"{explanation_dir}/{score_name}/{score_parser_name}")
    # score_parser.read(
    #     f"{explanation_dir}/{score_name}/{score_parser_name}"
    #     # f"{explanation_dir}/{score_parser_name}"
    # )
    # score_config = score_parser.items("score")
    # score_config = config_handler.section_to_dictionary(
    #     score_config, [",", "\n"]
    # )

    # score_configuration = {}
    # score_configuration["metric"] = metric
    # score_configuration["lines"] = score_config["lines"]
    # score_configuration["velocity"] = velocity

    # if is_reconstruction is True:

    #     score_configuration["epsilon"] = score_config["epsilon"]
    #     score_configuration["relative"] = relative
    #     score_configuration["percentage"] = percentage

    # lime_configuration = parser.items("lime")
    # lime_configuration = config_handler.section_to_dictionary(
    #     lime_configuration, [",", "\n"]
    # )

    # fudge_configuration = config_handler.section_to_dictionary(
    #     parser.items("fudge"), value_separators=[]
    # )

    # save_explanation_to = (
    #     f"{explanation_dir}/{score_name}/"
    #     # f"xai_{spectra_name.split('.')[0]}"
    # )
    # check.check_dir(save_explanation_to, exit_program=False)

    # explanation_runs = glob.glob(f"{save_explanation_to}/*/")

    # print(explanation_runs, save_explanation_to)

    # if len(explanation_runs) == 0:

    #     run = "000"

    # else:

    #     runs = [int(run.split("/")[-2]) for run in explanation_runs]
    #     run = f"{max(runs)+1:03d}"

    # save_explanation_to = f"{save_explanation_to}/{run}"
    # check.check_dir(f"{save_explanation_to}", exit_program=False)
    # number_processes = parser.getint("configuration", "jobs")
    # cores_per_worker = parser.getint("configuration", "cores_per_worker")

    # with mp.Pool(
    #     processes=number_processes,
    #     initializer=parallelExplainer.init_shared_data,
    #     initargs=(
    #         counter,
    #         wave,
    #         specobjid,
    #         spectra_to_explain,
    #         spectra_to_explain_shape,
    #         score_configuration,
    #         lime_configuration,
    #         fudge_configuration,
    #         model_dir,
    #         save_explanation_to,
    #         cores_per_worker,
    #     ),
    # ) as pool:

    #     pool.map(
    #         parallelExplainer.explain_anomalies,
    #         np.arange(spectra_to_explain_shape[0]),
    #     )

    # with open(
    #     f"{save_explanation_to}/{config_file_name}", "w", encoding="utf8"
    # ) as config_file:

    #     parser.write(config_file)
    # finish_time = time.time()
    # print(f"\nRun time: {finish_time - start_time:.2f}")
if __name__ == "__main__":
    main()