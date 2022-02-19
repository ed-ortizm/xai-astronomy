import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.lime_image import ImageExplanation
class TestMeWhy:
    def __init__(self, explanation: ImageExplanation, dtype: str):

        assert dtype in ["image", "spectra"]

        self.explanation = explanation

        if dtype == "spectra":
            self.galaxy = explanation.image[0, :, 0]
            self.segments = explanation.segments[0, :, 0]

        else:
            self.galaxy = explanation.image
            self.segments = explanation.segments

        self.dtype = dtype

    ###########################################################################
    def show_me(
        self,
        positive_only: bool=True,
        negative_only: bool=False,
        number_of_features: int=5,
        hide_rest: bool=False,
        minimum_weight: float=-np.inf,
        show_explanation: bool=False,
    ) -> np.array:

        image, mask = self.explanation.get_image_and_mask(
            label=self.explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=number_of_features,
            hide_rest=hide_rest,
            min_weight = minimum_weight
        )

        visual_explanation = mark_boundaries(image, mask)

        if show_explanation is True:

            plt.imshow(visual_explanation)

        return visual_explanation
    ###########################################################################
    def heatmap(self,
        show_map: bool=False,
        save_map: bool=False,
        save_to: str=".", galaxy_name: str="name", save_format: str=".png"
    ) -> None:


        heatmap = self.get_heatmap()
        #The visualization makes more sense if a symmetrical colorbar is used.
        min_max = np.abs(heatmap).max()

        if self.dtype == "image"

            plt.imshow(heatmap, cmap="RdBu", vmin=-min_max, vmax=min_max)

            plt.colorbar()

        else:
            self._heatmap_spectrum()

        if show_map is True:
            plt.show()

        if save_map is True:

            plt.savefig(f"{save_to}/{galaxy_name}HeatMapExp.{save_format}")
    ###########################################################################
    def _heatmap_spectrum(self):
        pass
    ###########################################################################
    def get_heatmap(self) -> np.array:

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.explanation.local_exp[ind])

        heatmap = np.vectorize(dict_heatmap.get)(self.segments)

        return heatmap
    ###########################################################################
    def segmentation(self, show_segmentation: bool=True):

        segmented_image = mark_boundaries(self.galaxy, self.segments)

        if show_segmentation is True:
            plt.imshow(segmented_image)
    ###########################################################################
###############################################################################
###############################################################################
class TellMeWhy:
    def __init__(self, explanation: ImageExplanation):

        self.explanation = explanation
        self.galaxy = explanation.image
        self.segments = explanation.segments

    ###########################################################################
    def show_me(
        self,
        positive_only: bool=True,
        negative_only: bool=False,
        number_of_features: int=5,
        hide_rest: bool=False,
        minimum_weight: float=-np.inf,
        show_explanation: bool=False,
    ) -> np.array:

        image, mask = self.explanation.get_image_and_mask(
            label=self.explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=number_of_features,
            hide_rest=hide_rest,
            min_weight = minimum_weight
        )

        visual_explanation = mark_boundaries(image, mask)

        if show_explanation is True:

            plt.imshow(visual_explanation)

        return visual_explanation
    ###########################################################################
    def heatmap(self,
        show_map: bool=False,
        save_map: bool=False,
        save_to: str=".", galaxy_name: str="name", save_format: str=".png"
    ) -> None:

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(self.segments)

        #The visualization makes more sense if a symmetrical colorbar is used.
        plt.imshow(
            heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max()
        )

        plt.colorbar()

        if show_map is True:
            plt.show()

        if save_map is True:

            plt.savefig(f"{save_to}/{galaxy_name}HeatMapExp.{save_format}")
    ###########################################################################
    def segmentation(self, show_segmentation: bool=True):

        segmented_image = mark_boundaries(self.galaxy, self.segments)

        if show_segmentation is True:
            plt.imshow(segmented_image)
###############################################################################
