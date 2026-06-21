import argparse
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import os
import time
import multiprocessing as mp

import numpy as np

from anomaly.constants import GALAXY_LINES
from astroExplain.spectra.utils import get_anomaly_score_name
import astroExplain.spectra.vae_parallel as vae_parallel

# Disable heavy multithreading in backends to prevent CPU lockups with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser_arg = argparse.ArgumentParser(description="Parallel VAE Explanations")
    parser_arg.add_argument("--config", type=str, default="vae_parallel_explain.ini")
    args = parser_arg.parse_args()

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_name = args.config
    parser.read(config_file_name)

    # 1. Directories
    spectra_dir = parser.get("directory", "spectra")
    models_dir = parser.get("directory", "models")
    bin_id = parser.get("common", "bin")
    explanation_dir = parser.get("directory", "explanations")

    # 2. File Names
    spectra_file = parser.get("file", "spectra")
    wave_file = parser.get("file", "wave")
    ids_file = parser.get("file", "ids")

    score_config = dict(parser.items("score"))

    score_name = get_anomaly_score_name(
        metric=score_config["metric"],
        velocity=int(score_config["filter"]),
        relative=parser.getboolean("score", "relative"),
        percentage=int(score_config["percentage"]),
    )

    # 3. File paths
    spectra_path = f"{explanation_dir}/{score_name}/{spectra_file}"
    wave_path = f"{spectra_dir}/{wave_file}"
    ids_path = f"{explanation_dir}/{score_name}/{ids_file}"
    model_path = f"{models_dir}/{bin_id}/winner"

    # 4. Output directories

    # build runtime name for explanations folder
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    segmentation = parser.get("lime", "segmentation")
    n_segments = parser.get("lime", "number_segments")
    n_samples = parser.get("lime", "number_samples")
    fudge_kind = parser.get("fudge", "kind_of_fudge")
    scale_factor = parser.get("fudge", "scale_factor")

    # Format: xai_"%Y%m%d%H%M"_uniform_110_samples_5000_scale_0.9
    xai_dir_name = (
        f"xai_{timestamp}_{segmentation}_{n_segments}_samples_"
        f"{n_samples}_{fudge_kind}_{scale_factor}"
    )
    # Final save path for explanations
    save_explanation_to = f"{explanation_dir}/{score_name}/{xai_dir_name}"
    os.makedirs(save_explanation_to, exist_ok=True)

    # Save a copy of the config file into this new directory for reproducibility!
    with open(
        f"{save_explanation_to}/{config_file_name}", "w", encoding="utf8"
    ) as config_file:
        parser.write(config_file)

    # 5. Load specobjids array
    target_specobjids = np.load(ids_path)

    number_processes = parser.getint("configuration", "jobs")

    score_config = {
        "metric": parser.get("score", "metric"),
        "filter": parser.getint("score", "filter"),
        "relative": parser.getboolean("score", "relative"),
        "percentage": parser.getint("score", "percentage"),
        "epsilon": parser.getfloat("score", "epsilon"),
        "lines": list(GALAXY_LINES.keys()),
    }

    lime_config = {
        "segmentation": parser.get("lime", "segmentation"),
        "number_segments": parser.getint("lime", "number_segments"),
        "number_samples": parser.getint("lime", "number_samples"),
        "batch_size": parser.getint("lime", "batch_size"),
        "progress_bar": parser.getboolean("lime", "progress_bar"),
        "distance_metric": parser.get("lime", "distance_metric"),
        "number_features": parser.getint("lime", "number_features"),
    }

    fudge_config = {
        "kind_of_fudge": parser.get("fudge", "kind_of_fudge"),
        "scale_factor": parser.getfloat("fudge", "scale_factor"),
        "same_noise": parser.getboolean("fudge", "same_noise"),
        "kernel_size": parser.getint("fudge", "kernel_size"),
        "sigma": parser.getfloat("fudge", "sigma"),
    }
    # 6. Launch Multiprocessing Pool
    print(f"Explanations for top 1% anomalies of {score_name}\n\n")

    start_time = time.perf_counter()
    with mp.Pool(
        processes=number_processes,
        initializer=vae_parallel.init_shared_data,
        initargs=(
            spectra_path,
            wave_path,
            ids_path,
            score_config,
            lime_config,
            fudge_config,
            model_path,
            save_explanation_to,  # Passes the newly created timestamped directory!
        ),
    ) as pool:

        # Map the targets to the worker function
        pool.map(vae_parallel.explain_anomaly, target_specobjids)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken for explanations: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
