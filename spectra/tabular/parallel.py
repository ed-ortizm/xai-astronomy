#! /usr/bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import sys
import time

###############################################################################
import lime
from lime import lime_tabular

###############################################################################
import numpy as np

###############################################################################
from astroxai.explainers.tabular import SpectraTabularExplainer
from autoencoders.variational.autoencoder import VAE
from anomaly.reconstruction import ReconstructionAnomalyScore
###############################################################################
ti = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("tabular.ini")
###############################################################################
if __name__ == __main__:

    print(f"Creating explainer")
    model_location = parser.get("directories", "model")
    model = VAE.load(model_location)
    mse = ReconstructionAnomalyScore(model).mse
    regressor = partial(mse, percentage=10)
    ###############################################################################
    train_data = np.load(parser.get("files", "train_data"))
    explainer_parameters = dict(parser.items("explainer"))


    explainer = SpectraTabularExplainer(
    train_data, explainer_parameters, regressor
    )

    #######################################################################
    def to_numpy_array(shared_array, array_shape):
        numpy_array = np.ctypeslib.as_array(shared_array)
        return numpy_array.reshape(array_shape)

    #######################################################################
    def init_worker(shared_array, array_shape):

        global spectra

        spectra = to_numpy_array(shared_array, array_shape)

    #######################################################################
    def explain_anomalies(spectra_indexes, number_features):

        print("\n" * 3)
        ###########################################################################
        explainer.explain_anomalies(
        anomalies=train_data[:10],
        number_features=[0],
        number_processes = 2
        )

    #######################################################################


        spectra_shape = anomalies.shape

        shared_spectra = RawArray(
                            ctypes.c_float,
                            anomalies.flatten()
                        )
        spectra_indexes = range(spectra_shape[0])

        starmap_parameters = product(spectra_indexes, number_features)

        with mp.Pool(
            processes=number_processes,
            initializer=init_worker,
            initargs=(shared_spectra, spectra_shape),
        ) as pool:

            results = pool.starmap(
                                    self._explain_anomaly,
                                    starmap_parameters
                                )

        return results

    ###########################################################################
    ###############################################################################
    ###########################################################################
    tf = time.time()
    print(f"Running time: {tf-ti:.2f} s")
    ################################################################################
