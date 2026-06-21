import argparse
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import multiprocessing as mp
import os
import time
import numpy as np

import astroExplain.spectra.latent_parallel as latent_parallel

# Disable heavy multithreading in backends
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser_arg = argparse.ArgumentParser(description="Parallel Latent Explanations")
    parser_arg.add_argument("--config", type=str, default="latent_parallel_explain.ini")
    args = parser_arg.parse_args()

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    config_file_name = args.config
    parser.read(config_file_name)

    # 1. Directories & Toggles
    spectra_dir = parser.get("directory", "spectra")
    models_dir = parser.get("directory", "models")
    latent_dir = parser.get("directory", "latent_models")
    bin_id = parser.get("common", "bin")
    model_type = parser.get("latent", "model_type")

    score_base_dir = parser.get("directory", "explanations")

    if not os.path.exists(score_base_dir):
        raise FileNotFoundError(f"Directory {score_base_dir} does not exist!")

    # 2. Build Output Directory String
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    segmentation = parser.get("lime", "segmentation")
    n_segments = parser.get("lime", "number_segments")
    n_samples = parser.get("lime", "number_samples")
    fudge_kind = parser.get("fudge", "kind_of_fudge")
    scale_factor = parser.get("fudge", "scale_factor")

    xai_dir_name = (
        f"xai_{timestamp}_{segmentation}_{n_segments}_samples_"
        f"{n_samples}_{fudge_kind}_{scale_factor}"
    )
    save_explanation_to = f"{score_base_dir}/{xai_dir_name}"
    os.makedirs(save_explanation_to, exist_ok=True)

    with open(
        f"{save_explanation_to}/{config_file_name}", "w", encoding="utf8"
    ) as config_file:
        parser.write(config_file)

    # 3. File Paths
    wave_path = f"{spectra_dir}/{parser.get('file', 'wave')}"
    anomalies_spectra_path = f"{score_base_dir}/{parser.get('file', 'spectra')}"
    anomalies_ids_path = f"{score_base_dir}/{parser.get('file', 'ids')}"

    # Path to VAE model
    vae_model_dir = f"{models_dir}/{bin_id}/winner"

    # Paths to Scikit-Learn models
    scaler_path = f"{latent_dir}/{parser.get('latent', 'scaler_model')}"

    if model_type == "iforest":
        sklearn_model_path = (
            f"{latent_dir}/iforest/{parser.get('latent', 'iforest_model')}"
        )
    elif model_type == "lof":
        sklearn_model_path = f"{latent_dir}/lof/{parser.get('latent', 'lof_model')}"
    else:
        raise ValueError("model_type in [latent] must be 'iforest' or 'lof'")

    # 4. Load specobjids
    target_specobjids = np.load(anomalies_ids_path)

    number_processes = parser.getint("configuration", "jobs")

    # Parse LIME configs safely to numbers/booleans
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

    print(f"Loaded {len(target_specobjids)} anomalies for {model_type.upper()}.")
    print(f"Saving explanations to: {save_explanation_to}")

    # 5. Launch Pool
    start_time = time.perf_counter()
    with mp.Pool(
        processes=number_processes,
        initializer=latent_parallel.init_shared_data,
        initargs=(
            anomalies_spectra_path,
            wave_path,
            anomalies_ids_path,
            lime_config,
            fudge_config,
            vae_model_dir,
            scaler_path,
            sklearn_model_path,
            model_type,  # Tell worker which wrapper to use!
            save_explanation_to,
        ),
    ) as pool:

        pool.map(latent_parallel.explain_anomaly, target_specobjids)

    end_time = time.perf_counter()
    dt = end_time - start_time
    print(f"Elapsed time: {dt/60:.2f} minutes")


if __name__ == "__main__":
    main()
