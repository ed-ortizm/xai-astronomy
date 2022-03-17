import numpy as np
###############################################################################
class SpectraSegmentation:
    """
    Class with different segmentation algorithms for spectra
    """
    ###########################################################################
    def __init__(self):
        pass
    ###########################################################################
    def uniform(self, spectra: np.array,
        number_segments: int=100
    ) -> np.array:
        """
        Divide spectra in bins of equal size

        INPUTS
            spectra: array with a single spectrum
            number_segments: define the width of each segment

        OUTPUTS
            segments: array with integers representing each segment
        """
        # number_segments = 100

        spectra = self.spectra_to_batch_image(spectra)
        size = spectra[0, 0, :, 0].size

        segment_size, residual = divmod(size, number_segments)

        # segments = np.empty(spectra[:, 0, :, 0].shape)
        segments = np.empty(spectra[:, :, :, 0].shape)

        for label_segment in range(number_segments):

            start_segment = label_segment*segment_size
            finish_segment = (label_segment+1)*segment_size

            segments[:, 0, start_segment:finish_segment] = label_segment

        if residual != 0:
            segments[:, 0, finish_segment:] = number_segments

        return segments.astype(int)
###########################################################################
    def spectra_to_batch_image(self, spectra):

        # If a 1D spec is passed
        if spectra.ndim == 1:
            # get (1, flux)
            gray_spectra = spectra[np.newaxis, ...]
            # get (1, flux, 3)
            spectra_image = gray2rgb(gray_spectra)
            # get (n_batch, 1, flux, 3)
            return spectra_image[np.newaxis]
        # array of spectra: (n_batch, flux)
        if spectra.ndim == 2:
            # get (n_bacth, flux, 3)
            gray_spectra = gray2rgb(spectra)
            # get (n_bacth, 1, flux, 3)
            return gray_spectra[:, np.newaxis, ...]
        # if already image pass to (n_batch, 1, flux, 3)
        if spectra.ndim == 3:
            return spectra[np.newaxis, ...]

        return spectra
    ###########################################################################
    def _update_dimension(self, array: np.array) -> np.array:

        if array.ndim == 1:
            return array[np.newaxis, ...]

        return array
    ###########################################################################
