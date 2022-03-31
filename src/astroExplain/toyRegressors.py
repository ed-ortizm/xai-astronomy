import numpy as np
from skimage.color import gray2rgb  # convert spectra to 3 channels

###############################################################################
class SpectraPlus:
    """
    Class to add all flexes values in an spectra
    """

    def __init__(self):
        pass

    def predict(self, spectra: np.array) -> np.array:

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
        Compute brightness of the image using as a baseline either
        the median or the mean value per channel.
    """

    def __init__(self, base_line: str = "median"):

        """
            base_line: either median or mean.
                Compute brightness of image-[median or mean]
        """

        assert (base_line == "median") or (base_line == "mean")

        self.base_line = base_line

    ###########################################################################
    def predict(self, image: np.array) -> np.array:
        """
            Compute normalized brightness of galaxy over base_line
            per channel

            INPUT
                image: 3D image or batch of 3D images

            OUTPUT
                predition: 2D array with brightness of images in the batch
                    shape --> (batch_size, 1)
        """

        # (height, size, channels) -> (batch_size, height, size, channels)
        image = self._update_dimension(image)

        base_line_per_channel = self.get_base_line_per_channel(image)
        
        image = image - base_line_per_channel

        prediction = np.sum(image, axis=(1, 2, 3))

        return prediction.reshape((-1, 1))

    ###########################################################################
    def get_base_line_per_channel(self, image: np.array) -> np.array:
        """
            Compute baseline per channel, either mean or median

            INPUT
                image: 3D image or batch of 3D images

            OUTPUT
                array with median or mean per channel,
                    keeping the dimensions of image
        """

        # axis=(1, 2) -> the weidth and height of the image
        if self.base_line == "median":
            return  np.median(image, axis=(1, 2), keepdims=True)

        elif self.base_line == "mean":
            return  np.mean(image, axis=(1, 2), keepdims=True)

    ###########################################################################
    def _update_dimension(self, image: np.array) -> np.array:

        if image.ndim == 3:
            return image[np.newaxis, ...]

        return image


###############################################################################
class CubePlus:
    def __init__(self, wave, line=6_562.8, delta=10, cube=True):

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

        neighborhood = (self.line - self.delta) < self.wave
        neighborhood *= self.wave < (self.line + self.delta)

        prediction = np.sum(image[..., neighborhood], axis=(1, 2, 3)).reshape(
            (-1, 1)
        )

        return prediction
