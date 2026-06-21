"""Parallel worker for generating LIME explanations on latent anomaly scores."""

import pickle
import joblib
import numpy as np
from functools import partial
import os

from autoencoders.ae import AutoEncoder
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explainer import LimeSpectraExplainer
from anomaly.latent import (
    LatentIForestAnomalyScore,
    LatentLOFAnomalyScore,
)


def init_shared_data(
    share_spectra_path,
    share_wave_path,
    share_ids_path,
    share_lime_config,
    share_fudge_config,
    share_vae_model_dir,
    share_scaler_path,
    share_sklearn_model_path,
    share_model_type,
    share_output_dir,
):
    """
    Initializes the Latent worker process.
    """
    global spectra_mmap
    global wave
    global specobjids
    global anomaly_score_function
    global lime_configuration
    global fudge_configuration
    global explainer
    global segmentation_fn
    global output_dir

    output_dir = share_output_dir
    lime_configuration = share_lime_config
    fudge_configuration = share_fudge_config

    # 1. Map data
    spectra_mmap = np.load(share_spectra_path, mmap_mode="r")
    specobjids = np.load(share_ids_path, mmap_mode="r")
    wave = np.load(share_wave_path)

    # 2. Load VAE
    ae_model = AutoEncoder(reload=True, reload_from=share_vae_model_dir)

    # 3. Load Sklearn Dependencies via Joblib
    scaler_model = joblib.load(share_scaler_path)
    sklearn_model = joblib.load(share_sklearn_model_path)

    if hasattr(sklearn_model, "n_jobs"):
        sklearn_model.n_jobs = 1

    # 4. Initialize the correct wrapper based on the config!
    if share_model_type == "iforest":
        anomaly_pipeline = LatentIForestAnomalyScore(
            encoder_model=ae_model.encode,
            scaler_model=scaler_model,
            iforest_model=sklearn_model,
        )
    elif share_model_type == "lof":
        anomaly_pipeline = LatentLOFAnomalyScore(
            encoder_model=ae_model.encode,
            scaler_model=scaler_model,
            lof_model=sklearn_model,
        )
    else:
        raise ValueError(f"Unknown model type: {share_model_type}")

    # Bind the .score method to our global pipeline variable
    anomaly_score_function = anomaly_pipeline.score

    # 5. Initialize LIME Explainer
    explainer = LimeSpectraExplainer(random_state=42)
    segmentation_algo = (
        SpectraSegmentation().kmeans
        if lime_configuration["segmentation"] == "kmeans"
        else SpectraSegmentation().uniform
    )
    segmentation_fn = partial(
        segmentation_algo, number_segments=lime_configuration["number_segments"]
    )


def explain_anomaly(target_specobjid):
    """
    Worker task: Explains a single anomaly and saves to disk.
    """
    save_path = f"{output_dir}/{target_specobjid}.pkl"
    if os.path.exists(save_path):
        return

    try:
        idx_spectrum = np.where(specobjids == target_specobjid)[0][0]
        galaxy = spectra_mmap[idx_spectrum]
        galaxy_2d = galaxy[np.newaxis, :]

        print(f"Explaining {target_specobjid}...", end="\r")

        explanation, _, _ = explainer.explain_instance(
            spectrum=galaxy_2d,
            classifier_fn=anomaly_score_function,
            segmentation_fn=segmentation_fn,
            fudge_parameters=fudge_configuration,
            explainer_parameters=lime_configuration,
        )

        with open(save_path, "wb") as file:
            pickle.dump(explanation, file)

    except Exception as e:
        print(f"\nError processing {target_specobjid}: {e}")
