import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.lime_image import ImageExplanation

###############################################################################
class TestMeWhy:
    def __init__(self,
        dtype: str,
        explanation: ImageExplanation,
        wave: np.array=None
    ):

        assert dtype in ["image", "spectra"]
        self.dtype = dtype

        if dtype == "spectra":
            self.galaxy = explanation.image[0, :, 0]
            self.segments = explanation.segments[0, :, 0]
            self.wave = wave

        else:
            self.galaxy = explanation.image
            self.segments = explanation.segments

        self.explanation = explanation

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
        symmetric_map: bool=None,
        save_to: str=".", galaxy_name: str="name", save_format: str=".png"
    ) -> None:


        heatmap = self.get_heatmap()

        if self.dtype == "image":
            min_max = np.abs(heatmap).max()

            plt.imshow(heatmap, cmap="RdBu", vmin=-min_max, vmax=min_max)

            plt.colorbar()

        else:
            self._plot_heatmap_spectrum(heatmap, symmetric_map)

        if show_map is True:
            plt.show()

        if save_map is True:

            plt.savefig(f"{save_to}/{galaxy_name}HeatMapExp.{save_format}")
    ###########################################################################
    def _plot_heatmap_spectrum(self, heatmap: np.array, symmetric_map: bool):

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array
        # so that we can stack points together easily to get the segments.
        # The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)

        points = np.array([self.wave, self.galaxy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots()

        if symmetric_map is True:

            value = np.abs(heatmap).max()
            vmin = -value
            vmax = value
            print(vmin, vmax)

        else:
            vmin = heatmap.min()
            vmax = heatmap.max()
            print(vmin, vmax)

        norm = plt.Normalize(vmin, vmax)

        lc = LineCollection(segments, cmap='RdBu', norm=norm)
        lc.set_array(heatmap)
        lc.set_linewidth(1.5)

        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

        ax.set_xlim(self.wave.min()-10, self.wave.max()+10)
        ax.set_ylim(self.galaxy.min()-2, self.galaxy.max()+2)

        plt.show()



    ###########################################################################
    def get_heatmap(self) -> np.array:

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.explanation.local_exp[ind])

        heatmap = np.vectorize(dict_heatmap.get)(self.segments)

        # Get average since line coloring requires the heatmap size to shrink
        heatmap = 0.5*(heatmap[:-1] + heatmap[1:])

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
