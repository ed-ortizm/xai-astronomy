"""
Doc string
"""

import numpy as np

# convert spectra to 3 channels
from skimage.color import gray2rgb


class SpecPlus:
    """
    Class to add all fluxes values in an spectra
    """

    def __init__(self):
        pass

    def predict(self, spectra: np.array) -> np.array:
        """
        imput

        return
        """

        # in case I pass a spectra with one dimension
        # this line converts 1D array to (1, n_wave, 3)
        # an image where each channel has the spectrun
        spectra = self.spectrum_to_image(spectra)
        # this is in case I pass a single spec
        # new axis added correponds to bach index
        spectra = self._update_dimension(spectra)

        assert spectra.ndim == 4

        # spectra-1 since spectra is normalized by the median
        # this way, values below the continua are detrimental
        # to the prediction
        prediction = np.sum(spectra - 1, axis=(1, 2, 3))
        # print(prediction.shape)

        return prediction.reshape((-1, 1))

    def spectrum_to_image(self, spectrum):
        """
        imput

        return
        """

        if spectrum.ndim == 1:

            gray_spectrum = spectrum[np.newaxis, ...]
            return gray2rgb(gray_spectrum)

        return spectrum

    def _update_dimension(self, spectra: np.array) -> np.array:

        if spectra.ndim == 3:
            return spectra[np.newaxis, ...]

        return spectra
