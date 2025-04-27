"""process base parallelism to to explain anomalies [spectra]"""
from functools import partial
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import pickle

import numpy as np

from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explainer import LimeSpectraExplainer
from anomaly.distance import DistanceAnomalyScore
from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from autoencoders.ae import AutoEncoder

def to_numpy_array(array: RawArray, array_shape: tuple = None) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    if array_shape is not None:
        return array.reshape(array_shape)

    return array


def init_shared_data(
    share_counter: mp.Value,
    share_wave: RawArray,
    share_specobjid: RawArray,
    share_anomalies: RawArray,
    data_shape: tuple,
    share_score_configuration: dict,
    share_lime_configuration: dict,
    share_fudge_configuration: dict,
    share_model_directory: str,
    share_output_directory: str,
    share_cores_per_worker: int,
) -> None:
    """
    Initialize worker to explain different

    PARAMETERS

        share_counter:
        share_anomalies:
        data_shape:
        share_model_directory:
        share_output_directory:

    """
    global counter
    global wave
    global specobjid
    global anomalies

    global score_configuration
    global lime_configuration
    global fudge_configuration

    global model_directory
    global save_explanation_to

    global cores_per_worker

    counter = share_counter
    wave = to_numpy_array(share_wave)
    specobjid = to_numpy_array(share_specobjid)

    anomalies = to_numpy_array(share_anomalies, data_shape)

    score_configuration = share_score_configuration
    lime_configuration = share_lime_configuration
    fudge_configuration = share_fudge_configuration

    model_directory = share_model_directory
    save_explanation_to = share_output_directory

    cores_per_worker = share_cores_per_worker

def explain_anomalies(_: int) -> None:
    """
    PARAMETERS
    """
    # Load reconstruction function
    model = AutoEncoder(reload=True, reload_from=model_directory)

    # Load anomaly score function
    is_reconstruction = (
        len({"lp", "mad", "mse"}.intersection({score_configuration["metric"]}))
        != 0
    )

    if is_reconstruction is True:

        anomaly = ReconstructionAnomalyScore(
            # reconstruct_function
            model.reconstruct,
            filter_parameters=FilterParameters(
                wave=wave,
                lines=score_configuration["lines"],
                velocity_filter=score_configuration["velocity"],
            ),
            reconstruction_parameters=ReconstructionParameters(
                percentage=score_configuration["percentage"],
                relative=score_configuration["relative"],
                epsilon=score_configuration["epsilon"],
            ),
        )

    else:
        anomaly = DistanceAnomalyScore(
            # reconstruct_function
            model.reconstruct,
            filter_parameters=FilterParameters(
                wave=wave,
                lines=score_configuration["lines"],
                velocity_filter=score_configuration["velocity"],
            ),
        )

    anomaly_score_function = partial(
        anomaly.score, metric=score_configuration["metric"]
    )

    # Set explainer instance
    # print(f"Set explainer and Get explanations", end="\n")
    explainer = LimeSpectraExplainer(random_state=0)
    segmentation_fn = None

    if lime_configuration["segmentation"] == "kmeans":

        segmentation_fn = SpectraSegmentation().kmeans

    elif lime_configuration["segmentation"] == "uniform":

        segmentation_fn = SpectraSegmentation().uniform

    segmentation_fn = partial(
        segmentation_fn, number_segments=lime_configuration["number_segments"]
    )

    ###########################################################################
    # Compute anomaly score
    with counter.get_lock():

        galaxy = anomalies[counter.value]
        # convert spectrum to gray image
        galaxy = galaxy[np.newaxis, :]

        specobjid_galaxy = specobjid[counter.value]

        print(f"[{counter.value}] Explain {specobjid_galaxy}", end="\r")

        counter.value += 1

    # Get explanations
    explanation = explainer.explain_instance(
        spectrum=galaxy,
        classifier_fn=anomaly_score_function,
        segmentation_fn=segmentation_fn,
        fudge_parameters=fudge_configuration,
        # hide_color=fudge_configuration["hide_color"],
        # amplitude=fudge_configuration["amplitude"],
        # mu=fudge_configuration["mu"],
        # std=fudge_configuration["std"],
        explainer_parameters=lime_configuration,
        # num_samples=lime_configuration["number_samples"],
        # batch_size=lime_configuration["batch_size"],
        # progress_bar=lime_configuration["progress_bar"],
        # distance_metric="cosine",
    )
    ###########################################################################
    with open(
        f"{save_explanation_to}/{specobjid_galaxy}.pkl",
        "wb",
        encoding=None,  # binary file :)
    ) as file:

        pickle.dump(explanation, file)
