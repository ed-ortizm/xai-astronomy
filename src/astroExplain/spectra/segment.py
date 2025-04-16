"""Segmentation algorithms for spectra"""
import numpy as np
# pylint: disable=E0611
from skimage.color import gray2rgb
from skimage.segmentation import slic

class SpectraSegmentation:
    """
    Class with different segmentation algorithms for spectra
    """

    def __init__(self):
        pass

    def kmeans(
        self,
        spectra: np.array,
        number_segments: int = 64,
        sigma: float = 3.0,
        compactness: float = 0.01,
    ) -> np.array:
        """
        Segment spectra according to flux distribution [color] using
        kmeans with algorithm skimage.segmentation.slic

        INPUTS
            spectra: array with a single spectrum
            number_segments: define the width of each segment
            sigma: shape of gaussian kernel to smooth spectrum
            compacness: smaller values make clusters according to
                flux values, while larger values make the segments
                similar to equally sized segments

        OUTPUTS
            segments: array with integers representing each segment
        """

        # if spectra.ndim == 1 --> (1, flux, 3)
        # if spectra.ndim == 2 --> (1, flux, 3)
        # if spectra.ndim == 3 --> spectra
        spectra = self.spectra_to_RGB(spectra)

        segments = slic(
            spectra,
            n_segments=number_segments,
            sigma=sigma,
            compactness=compactness,
        )

        return segments.astype(int)

    def uniform(
        self, spectra: np.array, number_segments: int = 64
    ) -> np.array:
        """
        Divide spectra in bins of equal size

        INPUTS
            spectra: array with a single spectrum
            number_segments: define the width of each segment

        OUTPUTS
            segments: array with integers representing each segment
        """

        # if spectra.ndim == 1 --> (1, flux, 3)
        # if spectra.ndim == 2 --> (1, flux, 3)
        # if spectra.ndim == 3 --> spectra
        spectra = self.spectra_to_RGB(spectra)
        size = spectra[0, :, 0].size

        segment_size, residual = divmod(size, number_segments)

        # set segments as a gray image
        segments = np.empty(spectra[:, :, 0].shape)

        for label_segment in range(number_segments):

            start_segment = label_segment * segment_size
            finish_segment = (label_segment + 1) * segment_size

            segments[0, start_segment:finish_segment] = label_segment

        if residual != 0:
            segments[0, finish_segment:] = number_segments

        return segments.astype(int)

    @staticmethod
    def spectra_to_RGB(spectra: np.array) -> np.array:
        """
        Convert spectra to RGB image, the height is set to 1 and
        each channel is a copy of the spectrum
        """

        # If a single spectrum is passed
        if spectra.ndim == 1:
            # get (1, flux)
            gray_spectra = spectra[np.newaxis, ...]
            # get (1, flux, 3)
            spectra_image = gray2rgb(gray_spectra)
            return spectra_image
        # gray image
        if spectra.ndim == 2:
            # get (1, flux, 3)
            spectra_image = gray2rgb(spectra)
            return spectra_image
        # if already image
        if spectra.ndim == 3:
            return spectra

        raise ValueError("Input array shape does not represent a spectrum")
