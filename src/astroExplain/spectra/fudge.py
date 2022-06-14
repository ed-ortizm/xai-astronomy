"""Functionality to fudge spectra"""
from astropy.convolution import Gaussian1DKernel, convolve
import numpy as np
from scipy.stats import norm

class Fudge:
    """Different methods to fudge a spectrum"""
    def __init__(self, image:np.array, segments:np.array):
        self.spectrum = image
        self.segments = segments

    def same(self)-> np.array:
        """The fudged spectrum is the same spectrum"""
        return self.spectrum.copy()

    def same_shape(self, kernel_size: int=3, sigma: float=0. )-> np.array:

        """
        Keep shape of spectrum and add white noise to it

        INPUTS

        kernel_size: size of gaussian kernel used to smoot the spectrum.
        sigma: standard deviation of white noise

        OUTPUT
        fudged_spectrum: filtered spectrum plus white noise
        """

        filtered_spectrum, _ = self.filter_noise(kernel_size)
        fudge_noise = self._white_noise(sigma=sigma)

        fudged_spectrum = filtered_spectrum + fudge_noise

        return fudged_spectrum


    def flat(self,
        continuum: float=1.,
        noise: str='spectrum',
        kernel_size: int=3,
        sigma: float=1
    )-> np.array:
        """
        Flat spectrum with different noise options. The noise can be
        zero, i.e, a complete flat spectrum. It can be the same noise
        in the spectrum or white noise according to input parameters

        INPUTS

        continuum: value of the continuum
        noise: options are {'spectrum', 'flat', 'white'}
            'spectrum': add the spectrum's noise to the continuum
            'flat': there is no noise added to the continuum
            'user': add whithe noise with mu=0 and sigma as passed
                in the arguments of this function
        kernel_size: size of gaussian kernel used to smoot the spectrum.
            Necessary when implementin noise='spectrum'. The noise
            will be the original image minus the the filtered image
            with the gaussian kernel
        sigma: standard deviation of white noise if noise is set
            to 'white'

        OUTPUT

        fugged_image: the fudged image

        """

        if noise == "spectrum":

            _, fudge_noise = self.filter_noise(kernel_size)
            fudged_spectrum = continuum + fudge_noise

        elif noise == "flat":

            fudged_spectrum = continuum * np.ones(self.spectrum.shape)

        elif noise == "white":

            fudge_noise = self._white_noise(sigma=sigma)
            fudged_spectrum = continuum + fudge_noise

        else:
            raise ValueError( "'noise' must be: 'spectrum', 'flat' or 'white'")

        return fudged_spectrum

    def filter_noise(self, kernel_size: int=3) -> tuple:
        """
        Filter noise on a spectrum with a gaussian kernel

        INPUT

        kernel_size: number of elements in gaussian kernel

        OUTPUT

        filtered_spectrum, noise:
            filtered_spectrum: spectrum with noise removed
            noise: spectrum's noise
        """
        kernel = Gaussian1DKernel(kernel_size)

        filtered_spectrum = convolve(self.spectrum, kernel, boundary="extend")
        noise = self.spectrum - filtered_spectrum

        return filtered_spectrum, noise

    def _white_noise(self, sigma=1.) -> np.array:
        """
        Generate array with white noise with the same shape of the
        image to fudge. This white noise has a median of zero

        INPUT
        sigma: standard deviation of the normal distribution

        OUTPUT
        noise: white noise
        """

        noise = np.random.normal(loc=0., scale=sigma, size=self.spectrum.shape)

        return noise

    def add_gaussians(
        self, amplitude: float = 1.0, std: float = 1.0
    ) -> np.array:
        """
        Create a fudged image adding an array of gaussians where each
        is gaussian is placed at the center of each segment and randomly
        assigned a positive or negative amplitude

        INPUTS
        amplitude: the amplitude of all gaussians
        std: common standard deviation to all gaussians

        OUTPUT
        image_fudged: image + array of gaussians
        """

        assert std > 0

        gaussians = self.get_gaussians(amplitude, std)

        image_fudged = self.spectrum.copy() + gaussians

        return image_fudged

    ###########################################################################
    def get_gaussians(
        self, amplitude: float = 1.0, std: float = 1.0
    ) -> (np.array, np.array):
        """
        Set array of gaussians to fudge the spectrum to explain. The
        sign of the amplitude  for each gaussian will be set randomly

        INPUTS
        amplitude: the amplitude of all gaussians
        std: common standard deviation to all gaussians

        OUTPUT
        gaussians: gray image representation of the array of gausians
        """
        number_gaussians = np.unique(self.segments).shape[0]
        number_pixels = self.spectrum[..., 0].size

        x = np.arange(number_pixels)
        centroids = self.get_centroids_of_segments()

        gaussians = np.zeros(shape=(1, number_pixels))

        amplitude *= np.random.choice([-1.0, 1.0], size=number_gaussians)
        for n in range(number_gaussians):

            mu = centroids[n]
            gaussians[0, :] += amplitude[n] * norm.pdf(x, mu, std)

        return gaussians.reshape((1, -1, 1))

    ###########################################################################
    def get_centroids_of_segments(self) -> np.array:

        """
        Get the index of the centroids for each segment

        OUTPUT
        centroids: centroids. indexes along the segments array
        """

        centroids = []

        for idx, segment_id in enumerate(np.unique(self.segments)):

            width = np.sum(self.segments == segment_id)

            if idx == 0:
                centroids.append(width / 2)

            else:
                centroids.append(width + centroids[idx - 1])

        centroids = np.array(centroids, dtype=int)

        return centroids

    ###########################################################################
    def fudge_with_mean(self) -> np.array:
        """
        Fudge image with mean value per channel per segmment

        OUTPUT
        image_fudged: original image + gaussian noise according
            to mu and std parameters
        """
        image_fudged = self.spectrum.copy()

        for segment_id in np.unique(self.segments):

            mask_segments = self.segments == segment_id

            mean_per_segment_per_channel = np.mean(
                self.spectrum[mask_segments], axis=(0, 1)
            )

            image_fudged[mask_segments] = mean_per_segment_per_channel

        return image_fudged
