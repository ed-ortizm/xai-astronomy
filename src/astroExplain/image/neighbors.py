import copy

import numpy as np
from skimage.segmentation import slic

###############################################################################
class Neighbors:
    """Get similar image by turning on and off superpixel using slic"""

    ###########################################################################
    def __init__(
        self,
        image: np.array,
        number_segments: int = 128,
        compactness: float = 32,
        sigma: float = 16,
        random_seed: int = None,
    ):

        """
        INPUT
        image:2D or 3D array containing the image
        number_segments: number of superpixels to split the image
        [from slic documentation]
        compactness:
            Balances color proximity and space proximity. Higher
            values give more weight to space proximity, making
            superpixel shapes more square/cubic. This parameter
            depends strongly on image contrast and on the shapes
            of objects in the image. We recommend exploring possible
            values on a log scale, e.g., 0.01, 0.1, 1, 10, 100,
            before refining around a chosen value
        sigma:
            Width of Gaussian smoothing kernel for pre-processing
            for each dimension of the image
        random_seed: if not None, the seed is set
        """

        self.image = image

        self.segments = slic(
            image,
            n_segments=number_segments,
            compactness=compactness,
            sigma=sigma,
        )

        if random_seed != None:
            np.random.seed(random_seed)

    ###########################################################################
    def get_neighbors(
        self,
        number_samples: int = 50,
        hide_color: float = 0.0,
    ) -> np.array:
        """
        Get samples aking to those generated by lime. The first
        element of the array is the original image

        INPUT
        number_samples: number of neighbors to sample
        hide_color: value to fill segments that
            "won't be cosidered" by the predictor. If None,
            it will fill each segment  with the mean value
            per channel
        OUTPUT
        neighbors: array with sampled neighbors. The first
            element of the array is the original image
        """

        fudged_galaxy = self.fudge_galaxy(hide_color)

        number_features = np.unique(self.segments).shape[0]

        on_off_batch_super_pixels = np.random.randint(
            0, 2, number_samples * number_features
        ).reshape((number_samples, number_features))

        # first row for the original image
        on_off_batch_super_pixels[0, :] = 1

        neighbors = []

        for on_off_super_pixels in on_off_batch_super_pixels:

            temp = copy.deepcopy(self.image)

            off_super_pixels = np.where(on_off_super_pixels == 0)[0]

            mask_off_superpixels = np.zeros(self.segments.shape).astype(bool)

            for off in off_super_pixels:
                mask_off_superpixels[self.segments == off] = True

            temp[mask_off_superpixels] = fudged_galaxy[mask_off_superpixels]

            neighbors.append(temp)

        return np.array(neighbors)

    ###########################################################################
    def fudge_galaxy(self, hide_color: float = 0.0) -> np.array:
        """
        Fudge image of galaxy to set pixel values of segments
        ignored in sampled neighbors

        INPUT
        hide_color: value to fill segments that
            "won't be cosidered" by the predictor. If None,
            it will fill each segment  with the mean value
            per channel
        OUTPUT
        fudged_galaxy: galaxy image with segments to ignore
            in neighbors set to hide_color
        """

        fudged_galaxy = self.image.copy()

        if hide_color == None:

            for segment in np.unique(self.segments):

                mask_segments = self.segments == segment

                mean_per_segment_per_channel = np.mean(
                    self.image[mask_segments], axis=(0, 1)
                )

                fudged_galaxy[mask_segments] = mean_per_segment_per_channel

        else:
            fudged_galaxy[:] = hide_color

        return fudged_galaxy

    ###########################################################################