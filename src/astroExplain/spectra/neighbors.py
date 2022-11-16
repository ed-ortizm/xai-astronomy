"""
Get neigbor spectra in parameter space as lime does to train an
explanier
"""
import copy
from typing import Callable

from skimage.color import gray2rgb
import numpy as np

from astroExplain.spectra.fudge import Fudge

###############################################################################
class SpectraNeighbors:

    "Generate neighboring spectra as LimeSpectraImageExplainer would"

    def __init__(
        self,
        random_seed: int = 1,
    ):

        """
        INPUT
        image: gray or RGB representation of a spectrum
        wave: wavelenght grid of the spectrum
        segmentation_function: segmentation funtion that has as
            input the image. Other parameters must be set via
            partial previously.
        random_seed: if not None, the seed is set
        """

        # check if image is gray or RGB

        np.random.seed(random_seed)

    ###########################################################################
    def get_neighbors(
        self,
        spectrum: np.array,
        fudge_parameters: dict,
        segmentation_function: Callable,
        number_samples: int = 50,
    ) -> np.array:
        """
        Get samples aking to those generated by lime. The first
        element of the array is the original image

        INPUT
        spectrum: spectrum to get neigbors of
        fudge_parameters: indicate the type of fudging, for instance:
            fudge_paramenters = {
            # same, same_shape, flat, with_mean, gaussians, scale
                "kind_of_fudge": "flat",
            # scale
                "scale_factor": 0.99,
            # flat
                "continuum": 1,
            # gaussians
                "amplitude":1.,
                "sigmas_in_segment": 8,
            # control-noise
                "same_noise": True,
                "kernel_size": 3,
                "sigma": 1
            }

        segmentation_function: Function to segment the spectra, can be:
            kmeans or uniform segmentation. Check Fudge class
        number_samples: number of neighbors to sample

        OUTPUT
            neighbors: array with sampled neighbors. The first
                element of the array is the original image
        """

        segments = segmentation_function(spectrum).reshape(-1)
        number_segments = np.unique(segments).size

        spectrum_fudged = self.fudge_spectrum(
            segments=segments,
            spectrum=spectrum,
            fudge_parameters=fudge_parameters,
        )

        on_off_batch_super_pixels = np.random.randint(
            0, 2, number_samples * number_segments
        ).reshape((number_samples, number_segments))

        # first row for the original image
        on_off_batch_super_pixels[0, :] = 1

        neighbors = []

        for on_off_super_pixels in on_off_batch_super_pixels:

            temp = copy.deepcopy(spectrum)

            off_super_pixels = np.where(on_off_super_pixels == 0)[0]

            mask_off_superpixels = np.zeros(segments.size).astype(bool)

            for off in off_super_pixels:
                mask_off_superpixels[segments == off] = True

            temp[mask_off_superpixels] = spectrum_fudged[mask_off_superpixels]

            neighbors.append(temp)

        return np.array(neighbors)

    @staticmethod
    def fudge_spectrum(
        spectrum: np.array, segments: np.array, fudge_parameters: dict
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
            fudged_spectrum: spectrum where all segments are perturbed
                to use when generating neigboring spectra.
        """

        fudge = Fudge(spectrum=spectrum, segments=segments)

        kind_of_fudge = fudge_parameters["kind_of_fudge"]

        if kind_of_fudge == "same":

            fudged_spectrum = fudge.same()

        elif kind_of_fudge == "scale":

            fudged_spectrum = fudge.scale(
                scale_factor=fudge_parameters["scale_factor"],
                same_noise=fudge_parameters["same_noise"],
                kernel_size=fudge_parameters["kernel_size"],
            )

        elif kind_of_fudge == "same_shape":

            fudged_spectrum = fudge.same_shape(
                kernel_size=fudge_parameters["kernel_size"],
                sigma=fudge_parameters["sigma"],
            )

        elif kind_of_fudge == "flat":

            fudged_spectrum = fudge.flat(
                continuum=fudge_parameters["continuum"],
                same_noise=fudge_parameters["same_noise"],
                kernel_size=fudge_parameters["kernel_size"],
                sigma=fudge_parameters["sigma"],
            )

        elif kind_of_fudge == "with_mean":

            fudged_spectrum = fudge.with_mean(
                same_noise=fudge_parameters["same_noise"],
                kernel_size=fudge_parameters["kernel_size"],
                sigma=fudge_parameters["sigma"],
            )

        elif kind_of_fudge == "gaussians":

            fudged_spectrum = fudge.gaussians(
                amplitude=fudge_parameters["amplitude"],
                sigmas_in_segment=fudge_parameters["sigmas_in_segment"],
                same_noise=fudge_parameters["same_noise"],
                kernel_size=fudge_parameters["kernel_size"],
                sigma=fudge_parameters["sigma"],
            )

        else:
            raise NotImplementedError(f"{kind_of_fudge} fudge not implemented")

        return fudged_spectrum
