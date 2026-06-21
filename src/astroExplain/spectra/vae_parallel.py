import pickle
import numpy as np
from functools import partial

# Assuming these are accessible in your environment
from autoencoders.ae import AutoEncoder
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explainer import LimeSpectraExplainer
from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters


def init_shared_data(
    share_spectra_path,
    share_wave_path,
    share_ids_path,
    share_score_config,
    share_lime_config,
    share_fudge_config,
    share_model_dir,
    share_output_dir,
):
    """
    Initializes the worker process. Loads the VAE and maps the data.
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

    # 1. Memory map the large arrays directly in the worker!
    spectra_mmap = np.load(share_spectra_path, mmap_mode="r")
    wave = np.load(share_wave_path)
    specobjids = np.load(share_ids_path, mmap_mode="r")

    # 2. Load VAE using your custom class
    ae_model = AutoEncoder(
        reload=True,
        reload_from=share_model_dir,
    )

    # 3. Initialize Reconstruction Score Function
    anomaly = ReconstructionAnomalyScore(
        reconstruct_function=ae_model.reconstruct,
        filter_parameters=FilterParameters(
            wave=wave,
            lines=share_score_config["lines"],
            velocity_filter=share_score_config["filter"],
        ),
        reconstruction_parameters=ReconstructionParameters(
            percentage=share_score_config["percentage"],
            relative=share_score_config["relative"],
            epsilon=share_score_config["epsilon"],
        ),
    )
    anomaly_score_function = partial(anomaly.score, metric=share_score_config["metric"])

    # 4. Initialize LIME Explainer
    explainer = LimeSpectraExplainer(random_state=42)

    segmentation_algo = None
    if lime_configuration["segmentation"] == "kmeans":
        segmentation_algo = SpectraSegmentation().kmeans
    else:
        segmentation_algo = SpectraSegmentation().uniform

    segmentation_fn = partial(
        segmentation_algo, number_segments=lime_configuration["number_segments"]
    )


def explain_anomaly(target_specobjid):
    """
    Worker task: Explains a single anomaly and saves to disk.
    """
    try:
        # Find the index of the requested specobjid
        idx_spectrum = np.where(specobjids == target_specobjid)[0][0]

        # Extract the specific spectrum (loads into memory from mmap)
        galaxy = spectra_mmap[idx_spectrum]
        galaxy_2d = galaxy[np.newaxis, :]

        print(f"Explaining {target_specobjid}...", end="\r")

        # Get explanations
        explanation, _, _ = explainer.explain_instance(
            spectrum=galaxy_2d,
            classifier_fn=anomaly_score_function,
            segmentation_fn=segmentation_fn,
            fudge_parameters=fudge_configuration,
            explainer_parameters=lime_configuration,
        )

        # Save to disk
        save_path = f"{output_dir}/{target_specobjid}.pkl"
        with open(save_path, "wb") as file:
            pickle.dump(explanation, file)

    except Exception as e:
        print(f"Error processing {target_specobjid}: {e}")
