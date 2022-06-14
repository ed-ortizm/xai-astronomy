"""Functionality to fudge spectra"""
import numpy as np
from scipy.stats import norm

class Fudge:
    """Different methods to fudge a spectrum"""
    def __init__(self, image:np.array, segments:np.array):
        self.image = image
        self.segments = segments
    ###########################################################################
    def fudge_spectrum(
        self, hide_color: str, amplitude: float = 1.0, mu=0, std=1.0
    ) -> np.array:
        """
        Fudge image of galaxy to set pixel values of segments
        ignored in sampled neighbors

        INPUT
            hide_color: value or method to perturbe segments in neigboring
                spectra.
                If "mean", each segment of the fudged image will contain
                    the mean value of the fluxes in that segment.
                If "noise", each segment of the fudged image will contain
                    the flux plus white noise.
                If "gaussian", each segment of the fudged image will contain
                    the flux plus a gaussian per segment. The absolute value
                    of the gaussians' amplitude wil be determined by the
                    variable amplitude. The standard deviation will be
                    determined by the variable std. The sign of gausians'
                    amplitude wil be randomly set.
                If numeric value, each segment of the fudged image will
                contain that the passed numeric value
            amplitude: amplitude of gaussians
            mu: mean of white-noise
            std: standard deviation of gaussians or white-noise
        OUTPUT
            image_fudged: spectrum where all segments are perturbed
                to use when generating neigboring spectra.
        """

        if hide_color == "mean":

            image_fudged = self.fudge_with_mean()

        elif hide_color == "noise":

            image_fudged = self.add_white_noise(mu, std)

        elif hide_color == "gaussians":

            image_fudged = self.add_gaussians(amplitude, std)

        else:

            is_a_number = isinstance(hide_color, float)
            is_a_number |= isinstance(hide_color, int)

            if is_a_number is True:
                # Fudge image with hide_color value on all pixels
                image_fudged = np.ones(self.image.shape) * hide_color

            else:

                raise ValueError(
                    "hide_color: 'mean', 'noise', 'gaussians' or numeric value"
                )

        return image_fudged

    ###########################################################################
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

        image_fudged = self.image.copy() + gaussians

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
        number_pixels = self.image[..., 0].size

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
        image_fudged = self.image.copy()

        for segment_id in np.unique(self.segments):

            mask_segments = self.segments == segment_id

            mean_per_segment_per_channel = np.mean(
                self.image[mask_segments], axis=(0, 1)
            )

            image_fudged[mask_segments] = mean_per_segment_per_channel

        return image_fudged

    ###########################################################################
    def add_white_noise(self, mu=1, std=0.1) -> np.array:
        """
        Fudge image with gaussian noise per channel per segmment

        INPUT
        mu: mean of the normal distribution in case hide color
            is set to "normal"
        std: standard deviation of the normal distribution in
            case hide color is set to "normal"

        OUTPUT
        image_fudged: original image + gaussian noise according
            to mu and std parameters
        """

        # image_fudged = self.image.copy()

        image_fudged = np.random.normal(mu, std, size=self.image.shape)

        return image_fudged
