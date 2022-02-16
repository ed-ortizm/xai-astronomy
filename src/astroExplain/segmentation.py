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
        # number_segments: int=10
    ) -> np.array:
        """
        Divide spectra in bins of equal size

        INPUTS
            spectra: array with a single spectrum
            number_segments: define the width of each segment

        OUTPUTS
            segments: array with integers representing each segment
        """
        number_segments = 10

        spectra = self._update_dimension(spectra)
        size = spectra.shape[1]

        segment_size, residual = divmod(size, number_segments)

        segments = np.empty(spectra.shape)

        for label_segment in range(number_segments):

            start_segment = label_segment*segment_size
            finish_segment = (label_segment+1)*segment_size

            segments[0, start_segment:finish_segment] = label_segment

        if residual != 0:
            segments[0, finish_segment:] = number_segments

        return segments.astype(int)
    ###########################################################################
    def _update_dimension(self, array: np.array) -> np.array:

        if array.ndim == 1:
            return array[np.newaxis, ...]

        return array
    ###########################################################################
