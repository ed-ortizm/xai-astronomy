import numpy as np
from skimage.color import gray2rgb # convert spectra to 3 channels
###############################################################################
class SpectraPlus:
    """
    Class to add all flexes values in an spectra
    """

    def __init__(self):
        pass

    def predict(self, spectra: np.array) -> np.array:

        spectra = self.spectrum_to_image(spectra)
        spectra = self._update_dimension(spectra)

        assert spectra.ndim == 4

        # spectra-1 since spectra is normalized by the median
        # this way, values below the continua are detrimental
        # to the prediction
        prediction = np.sum(spectra - 1, axis=(1, 2, 3))
        # print(prediction.shape)

        return prediction.reshape((-1,1))

    ###########################################################################
    def spectrum_to_image(self, spectrum):

        if spectrum.ndim == 1:

            gray_spectrum = spectrum[np.newaxis, ...]
            return gray2rgb(gray_spectrum)

        return spectrum
    ###########################################################################
    def _update_dimension(self, spectra: np.array) -> np.array:

        if spectra.ndim == 3:
            return spectra[np.newaxis, ...]

        return spectra
###############################################################################
class GalaxyPlus:
    """
    Class to add all pixel values in an image
    """

    def __init__(self):
        pass

    def predict(self, image: np.array) -> np.array:

        image = self._update_dimension(image)
        mean_per_channel = self.get_mean_per_channel(image)

        # predict and normalize
        prediction = np.sum(image-mean_per_channel, axis=(1, 2, 3))

        return prediction.reshape((-1, 1))
    ###########################################################################
    def get_mean_per_channel(self, image: np.array) -> np.array:

        mean_per_channel = np.mean(image, axis=(1,2), keepdims=True)

        return mean_per_channel
    ###########################################################################

    def _update_dimension(self, image: np.array) -> np.array:

        if image.ndim == 3:
            return image[np.newaxis, ...]

        return image
###############################################################################
class CubePlus:

    def __init__(self,
        wave,
        line=6_562.8,
        delta=10,
        cube=True
        ):

        self.wave = wave
        self.line = line
        self.delta = delta
        self.cube = cube

    def predict(self, image):

        if image.ndim == 3:
            image = image.reshape((1,) + image.shape)

        if not self.cube:
            prediction = np.sum(image, axis=(1, 2, 3)).reshape((-1, 1))
            return prediction


        neighborhood = (self.line-self.delta) < self.wave
        neighborhood *= self.wave < (self.line + self.delta)

        prediction = np.sum(
            image[..., neighborhood],
            axis=(1, 2, 3)
            ).reshape((-1, 1))

        return prediction
