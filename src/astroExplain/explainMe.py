import copy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

import numpy as np
from skimage.segmentation import mark_boundaries
from lime.lime_image import ImageExplanation


###############################################################################
class TellMeWhyImage:
    def __init__(self, explanation: ImageExplanation):
        """
            INPUT
            explanation: an explanation generated by lime_image explainer.
        """

        self.explanation = explanation
        self.galaxy = explanation.image
        self.segments = explanation.segments

    ###########################################################################
    def get_neighbors(self,
        number_samples: int=50,
        hide_color: float=0.,
    )-> np.array:
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

        np.random.seed(0)

        number_features = np.unique(self.segments).shape[0]

        on_off_batch_super_pixels = np.random.randint(0, 2,
            number_samples * number_features
        ).reshape((number_samples, number_features))

        # first row for the original image, that is,
        # all super_pixels are on
        on_off_batch_super_pixels[0, :] = 1

        neighbors = []

        for on_off_super_pixels in on_off_batch_super_pixels:

            temp = copy.deepcopy(self.galaxy)

            off_super_pixels = np.where(
                on_off_super_pixels == 0
            )[0]

            mask = np.zeros(self.segments.shape).astype(bool)

            for off in off_super_pixels:
                mask[self.segments == off] = True

            temp[mask] = fudged_galaxy[mask]

            neighbors.append(temp)

        return np.array(neighbors)
    ###########################################################################
    def fudge_galaxy(self, hide_color: float = 0.) -> np.array:
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

        fudged_galaxy = self.galaxy.copy()

        if hide_color == None:

            for segment in np.unique(self.segments):

                mask_segments = self.segments == segment

                mean_per_segment_per_channel = np.mean(
                    self.galaxy[mask_segments], axis=(0, 1)
                )

                fudged_galaxy[mask_segments] = mean_per_segment_per_channel

        else:
            fudged_galaxy[:] = hide_color

        return fudged_galaxy
    ###########################################################################
    def show_me(
        self,
        positive_only: bool = True,
        negative_only: bool = False,
        number_of_features: int = 5,
        hide_rest: bool = False,
    ) -> np.array:

        """
            Explore super pixels with the largest possitive
            and negative impact to the prediction

            INPUT
            positive_only: if True, retrieves segments associated to
                positive weights of the explanation
            negative_only: if positive_only is False and this is True,
                retrieves segments associated to negative weights of
                the explanation
            number_of_features: example: 6, then it will get the six
                segments with the largest inpat to the anomaly score.
                If None, it will consider all the segments
            hide_rest: If True, it sets to zero the rest of the super
                pixel

            OUTPUT
            visual_explanation: array with boundaries highlighting
            relevant super pixels of the explanation
        """

        #######################################################################
        # check if user inputs right combination of bool values
        bad_input = (positive_only is True) and (negative_only is True)

        if bad_input is True:

            raise ValueError(
                f"positive_only and negative_only cannot be true"
                f"at the same time."
            )
        #######################################################################
        if number_of_features == None:
            # set to the total number of segments
            number_of_features = np.unique(self.segments).size

        #######################################################################
        image, mask = self.explanation.get_image_and_mask(
            label=self.explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=number_of_features,
            min_weight=-np.inf,
            hide_rest=hide_rest,
        )

        visual_explanation = mark_boundaries(image, mask)

        return visual_explanation

    ###########################################################################
    def get_heatmap(self) -> np.array:
        """
            Get heatmap of explanation for galaxy

            OUTPUT
            heatmap: 2 D array where pixel take the values of the
                explanation weights
        """

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(self.segments)

        return heatmap
    ###########################################################################
    def get_segmented_image(self) -> np.array:
        """
            Get image with super pixels highlited. It is the image, but
            in the interpretable representations of the superpixel.

            OUTPUT
            segmented_image: array highlighting super pixels
        """

        segmented_image = mark_boundaries(self.galaxy, self.segments)

        return segmented_image

###############################################################################
