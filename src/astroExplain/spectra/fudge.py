"""Functionality to fudge spectra"""
from astropy.convolution import Gaussian1DKernel, convolve
import numpy as np
from scipy.stats import norm


class Fudge:
    """Different methods to fudge a spectrum"""

    def __init__(self, spectrum: np.array, segments: np.array):
        self.spectrum = spectrum
        self.segments = segments

    def same(self) -> np.array:
        """The fudged spectrum is the same spectrum"""
        return self.spectrum.copy()

    def same_shape(self, kernel_size: int = 3, sigma: float = 0.0) -> np.array:

        """
        Keep shape of spectrum and add white noise to it

        INPUTS

        kernel_size: size of gaussian kernel used to smoot the spectrum.
        sigma: standard deviation of white noise. If sigma is zero, it
            returns the sahpe of the spectrum

        OUTPUT
        fudged_spectrum: filtered spectrum plus white noise
        """

        filtered_spectrum, _ = self.filter_noise(kernel_size)
        fudge_noise = self._white_noise(sigma=sigma)

        fudged_spectrum = filtered_spectrum + fudge_noise

        return fudged_spectrum

    def scale(self,
        scale_factor: float,
        same_noise: bool = True,
        kernel_size: int=3,
        # sigma: float = 0
    ) -> np.array:

        """
        Fudge spectrum by scaling original spectrum and keeping
        the same median. This fudging allows to reduce emission
        or absorption lines keeping the continuum and reshape
        the continuun as well. The scaling is as follows:
        spec = (spec-np.median(spec)) * scale_factor + np.median(spec)

        INPUT

        scale_factor: positive number to scale the spectrum
        same_noise: if True, add the spectrum's noise to the mean value
            of each segment. Otherwise add white noise, according to
            sigma. If sigma is zero, there is no noise.
        sigma: standard deviation of white noise if same_noise is False

        OUTPUT
        fudged_spectrum: original spectrum but scaled keeping
            its median
        """

        if same_noise is True:
            # only re-scale signal
            spectrum, fudge_noise = self.filter_noise(kernel_size)

            fudged_spectrum = (spectrum - np.nanmedian(spectrum))*scale_factor
            fudged_spectrum += np.nanmedian(spectrum)

            fudged_spectrum += fudge_noise

        else:
            # re-scale signal to noise
            spectrum = self.spectrum.copy()

            fudged_spectrum = (spectrum - np.nanmedian(spectrum))*scale_factor
            fudged_spectrum += np.nanmedian(spectrum)




        return fudged_spectrum

    def with_mean(self,
        same_noise: bool = True, kernel_size: int=3, sigma: float = 0
    ) -> np.array:

        """
        Fudge spectrum with mean value per channel per segmment
        and either same noise in original segment or white noise

        INPUT

        same_noise: if True, add the spectrum's noise to the mean value
            of each segment. Otherwise add white noise, according to
            sigma. If sigma is zero, there is no noise.
        sigma: standard deviation of white noise if same_noise is False

        OUTPUT
        fudged_spectrum: original image + gaussian noise according
            to mu and std parameters
        """

        fudged_spectrum = self.spectrum.copy()

        for segment_id in np.unique(self.segments):

            mask_segments = self.segments == segment_id

            mean_per_segment = np.mean(
                self.spectrum[mask_segments] #, axis=(0, 1)
            )

            fudged_spectrum[mask_segments] = mean_per_segment

        if same_noise is True:

            _, fudge_noise = self.filter_noise(kernel_size)

        else:

            fudge_noise = self._white_noise(sigma)

        fudged_spectrum += fudge_noise

        return fudged_spectrum

    def flat(
        self,
        continuum: float = 1.0,
        same_noise: bool = True,
        kernel_size: int = 3,
        sigma: float = 1,
    ) -> np.array:
        """
        Flat spectrum with different noise options. The noise can be
        zero, i.e, a complete flat spectrum. It can be the same noise
        in the spectrum or white noise according to input parameters

        INPUTS

        continuum: value of the continuum
        same_noise: if True, add the spectrum's noise to the continuum.
            otherwise add white noise to the continuum, according
            to sigma. If sigma is zero, there is no noise in the
            flat continuum
        kernel_size: size of gaussian kernel used to smoot the spectrum.
            Necessary when implementin noise='spectrum'. The noise
            will be the original image minus the the filtered image
            with the gaussian kernel
        sigma: standard deviation of white noise if same_noise is False

        OUTPUT

        fugged_image: the fudged image

        """

        if same_noise is True:

            _, fudge_noise = self.filter_noise(kernel_size)
            fudged_spectrum = continuum + fudge_noise

        else:

            fudge_noise = self._white_noise(sigma=sigma)
            fudged_spectrum = continuum + fudge_noise

        return fudged_spectrum

    def filter_noise(self, kernel_size: int = 3) -> tuple:
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

        #spectrum = self.spectrum[0, :, 0]
        #filtered_spectrum = convolve(spectrum, kernel, boundary="extend")
        filtered_spectrum = convolve(self.spectrum, kernel, boundary="extend")

        noise = self.spectrum - filtered_spectrum

        #filtered_spectrum = gray2rgb(filtered_spectrum.reshape(1, -1))
        #noise = gray2rgb(noise.reshape(1, -1))

        return filtered_spectrum, noise

    def _white_noise(self, sigma=1.0) -> np.array:
        """
        Generate array with white noise with the same shape of the
        image to fudge. This white noise has a median of zero

        INPUT
        sigma: standard deviation of the normal distribution

        OUTPUT
        noise: white noise
        """

        noise = np.random.normal(
            loc=0.0, scale=sigma, size=self.spectrum.shape
        )

        return noise

    def gaussians(self,
        amplitude: float = 1.0,
        sigmas_in_segment: int = 8,
        same_noise: bool = True,
        kernel_size: int = 3,
        sigma: float = 1.,
    ) -> np.array:
        """
        Create a fudged image adding an array of gaussians where each
        is gaussian is placed at the center of each segment and randomly
        assigned a positive or negative amplitude

        INPUTS
        amplitude: absolute value of the amplitude for all gaussians
        same_noise: if True, add the spectrum's noise to the continuum.
            otherwise add white noise to the continuum, according
            to sigma. If sigma is zero, there is no noise in the
            flat continuum
        kernel_size: size of gaussian kernel used to smoot the spectrum.
            Necessary when implementin noise='spectrum'. The noise
            will be the original image minus the the filtered image
            with the gaussian kernel
        sigma: standard deviation of white noise if same_noise is False

        OUTPUT
        fudged_spectrum: spectrum + noise + array of gaussians
        """

        # Get noise
        if same_noise is True:

            fudged_spectrum = self.spectrum.copy()

        else:

            fudged_spectrum, _ = self.filter_noise(kernel_size)
            fudge_noise = self._white_noise(sigma=sigma)
            fudged_spectrum += fudge_noise


        gaussians = self.get_gaussians(amplitude, sigmas_in_segment)

        fudged_spectrum += gaussians

        return fudged_spectrum

    def get_gaussians(
        self, amplitude: float = 1.0, sigmas_in_segment: int = 8
    ) -> np.array:
        """
        Set array of gaussians to fudge the spectrum to explain. The
        sign of the amplitude  for each gaussian will be set randomly

        INPUTS
        amplitude: the amplitude of all gaussians
        sigmas_in_segment: number of times the standard deviation
            of the gaussian fits in the segment

        OUTPUT
        gaussians: gray image representation of the array of gausians
        """
        # f_gaussian controls the spread of the gaussian but not
        # the amplitude
        f_gaussian = lambda x, mu, sigma: np.exp(-0.5*((x-mu)/sigma)**2)

        number_gaussians = np.unique(self.segments).shape[0]
        number_pixels = self.spectrum.size

        x = np.arange(number_pixels)
        mus, sigmas = self.get_mus_and_sigmas(sigmas_in_segment)

        gaussians = np.zeros(shape=(number_pixels))

        amplitudes = amplitude * np.ones(number_gaussians)

        for n in range(number_gaussians):

            mu = mus[n]
            sigma = sigmas[n]
            amplitude = amplitudes[n]

            if n == number_gaussians-1 :

                if sigma < 0.5*sigmas[n-1]:
                
                    amplitude = 0

            gaussians += amplitude * f_gaussian(x, mu, sigma)

        return gaussians

    ###########################################################################
    def get_mus_and_sigmas(self, sigmas_in_segment: int=8) -> np.array:

        """
        Get the index of the mus for each segment and the sigma of each gaussian

        INPUTS
        sigmas_in_segment: number of times the standard deviation
            of the gaussian fits in the segment

        OUTPUT
        mus, sigmas: centroids indexes along the segments array and sigma
            of gaussian
        """

        mus = []
        sigmas = []

        for idx, segment_id in enumerate(np.unique(self.segments)):

            width = np.sum(self.segments == segment_id)
            sigmas.append(width/sigmas_in_segment)

            if idx == 0:
                mus.append(width / 2)

            else:
                mus.append(width + mus[idx - 1])

        mus = np.array(mus, dtype=int)
        sigmas = np.array(sigmas)

        return mus, sigmas

    ###########################################################################
