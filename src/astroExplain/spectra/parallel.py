"""process base parallelism to to explain anomalies [spectra]"""
from functools import partial
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import pickle

import numpy as np

from anomaly.reconstruction import ReconstructionAnomalyScore
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explainer import LimeSpectraExplainer
# from sdss.utils.managefiles import FileDirectory

###############################################################################
def to_numpy_array(array: RawArray, array_shape: tuple = None) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    if array_shape is not None:
        return array.reshape(array_shape)

    return array


###############################################################################
def init_shared_data(
    share_counter: mp.Value,
    share_wave: RawArray,
    share_specobjid: RawArray,
    share_anomalies: RawArray,
    data_shape: tuple,
    share_score_configuration: dict,
    share_lime_configuration: dict,
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

    global model_directory
    global save_explanation_to

    global cores_per_worker
    global session

    counter = share_counter
    wave = to_numpy_array(share_wave)
    specobjid = to_numpy_array(share_specobjid)

    anomalies = to_numpy_array(share_anomalies, data_shape)

    score_configuration = share_score_configuration
    lime_configuration = share_lime_configuration

    model_directory = share_model_directory
    save_explanation_to = share_output_directory

    cores_per_worker = share_cores_per_worker


###############################################################################
# def explain_anomalies(number_anomaly: int) -> None:
#     # def explain_anomalies() -> None:
#     """
#     PARAMETERS
#     """
#     ###########################################################################
#     import tensorflow as tf
#     from autoencoders.ae import AutoEncoder
#
#     # set the number of cores to use per model in each worker
#     jobs = cores_per_worker
#     config = tf.compat.v1.ConfigProto(
#         intra_op_parallelism_threads=jobs,
#         inter_op_parallelism_threads=jobs,
#         allow_soft_placement=True,
#         device_count={"CPU": jobs},
#     )
#     session = tf.compat.v1.Session(config=config)
#     ###########################################################################
#     # Load reconstruction function
#     # print(f"Load reconstruction function", end="\n")
#
#     model = AutoEncoder(reload=True, reload_from=model_directory)
#     reconstruct_function = model.reconstruct
#
#     ###########################################################################
#     # Load anomaly score function
#     # print(f"Load anomaly score function", end="\n")
#
#     anomaly = ReconstructionAnomalyScore(
#         reconstruct_function,
#         wave,
#         lines=score_configuration["lines"],
#         velocity_filter=score_configuration["velocity"],
#         percentage=score_configuration["percentage"],
#         relative=score_configuration["relative"],
#         epsilon=score_configuration["epsilon"],
#     )
#
#     anomaly_score_function = partial(
#         anomaly.score, metric=score_configuration["metric"]
#     )
#     ###########################################################################
#     # Set explainer instance
#     # print(f"Set explainer and Get explanations", end="\n")
#     explainer = LimeSpectraExplainer(random_state=0)
#
#     segmentation_fn = SpectraSegmentation().uniform
#     segmentation_fn = partial(
#         segmentation_fn, number_segments=lime_configuration["number_segments"]
#     )
#
#     ###########################################################################
#     # Compute anomaly score
#     with counter.get_lock():
#
#         galaxy = anomalies[counter.value]
#         # convert spectrum to gray image
#         galaxy = galaxy[np.newaxis, :]
#
#         specobjid_galaxy = specobjid[counter.value]
#
#         print(f"[{counter.value}] Explain", end="\r")
#
#         counter.value += 1
#
#     # Get explanations
#     explanation = explainer.explain_instance(
#         image=galaxy,
#         classifier_fn=anomaly_score_function,
#         segmentation_fn=segmentation_fn,
#         hide_color=lime_configuration["hide_color"],
#         amplitude=lime_configuration["amplitude"],
#         mu=lime_configuration["mu"],
#         std=lime_configuration["std"],
#         num_samples=lime_configuration["number_samples"],
#         batch_size=lime_configuration["batch_size"],
#         progress_bar=lime_configuration["progress_bar"],
#         # distance_metric="cosine",
#     )
#     ###########################################################################
#     save_name = f"{specobjid_galaxy}"
#
#     with open(f"{save_explanation_to}/{save_name}.pkl", "wb") as file:
#
#         pickle.dump(explanation, file)
#     ###########################################################################
#     session.close()
