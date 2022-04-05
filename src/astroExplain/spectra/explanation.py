###############################################################################
class TellMeWhySpec:
    ###########################################################################
    """
        Class to illustrate lime_image explanations of anomalous spectra
        where a spectrum is took as a (1, number_of_fluxes) gray image
        and the superpixel are collections of pixels
    """
    ###########################################################################
    def __init__(self, explanation: ImageExplanation, wave: np.array):
        """
            INPUT
            explanation: an explanation generated by lime_image explainer.
            wave: array with the wavelengts associated to the spectrum
        """

        self.galaxy = explanation.image[0, :, 0]
        self.segments = explanation.segments[0, :]
        self.wave = wave

        self.explanation = explanation

    ###########################################################################
    def get_full_explanation(
        self,
        drop_fraction: float = 0.1,
        figure_size: tuple=(10, 5),
        save: bool = False,
        save_to: str = ".",
        galaxy_name: str = "name",
        save_format: str = "png",
    ) -> (plt.Figure, plt.Axes):
        """
            This method allows to see the spectrum and the normalized
            explanation weights pixel by pixel in two subplots

            INPUT
            drop_fraction: indicates the width of the band to ignore
                segments with explanation weights inside of this band.
                For instance, if it is 0.1, then weights with absolute
                values smaller to 0.1*np.abs(weights).max() would be
                ignore
            figure_size: tuple with the size of the figure
            save: If True, the image would be save to hard drive
            save_to: path to store the image
            galaxy_name: identification of the spectrum
            save_format: format to store the image

            OUTPUT
            (fig, ax): of the explanation
        """
        #######################################################################

        heatmap = self.get_heatmap()
        # smooth noise
        heatmap -= np.median(heatmap)
        # normalize heatmap
        heatmap *= 1/np.abs(heatmap).max()
        # smooth noise with drop_fraction
        heatmap = np.where(np.abs(heatmap) < drop_fraction, 0, heatmap)


        fig, axs = plt.subplots(
            nrows=2, ncols=1,
            sharex=True,
            figsize=figure_size
        )

        axs[0].plot(self.wave, self.galaxy, color="black")

        axs[1].plot(self.wave, heatmap, color="black")

        axs[1].plot(self.wave, np.zeros(heatmap.shape),
            color="blue", linestyle='dashed'
        )
        # axs[1].set_ylim([-1,1])

        if save is True:

            fig.savefig(f"{save_to}/{galaxy_name}_explanation.{save_format}")

        return fig, axs
    ###########################################################################
    def show_me(
        self,
        show_all: bool = False,
        show_positive_only: bool = False,
        show_negative_only: bool = False,
        number_of_features: int = None,
        drop_fraction: float = 0.2,
        save: bool = False,
        save_to: str = ".",
        galaxy_name: str = "name",
        save_format: str = "png",
    ) -> np.array:
        #######################################################################
        """
            Show segments that contribute to the anomaly score either
            positive, negative or both. Allows to set a threshold band
            on the explanations' weights to ignore segments of the
            spectrum with a negligible contribution to the inference

            INPUT
            show_all: if True, it shows segments with positive and
                negative contribution to the anomaly score
            show_positive_only: if True and show_all is False, it
                shows segments with positive contribution to the
                anomaly score
            show_negative_only: if True and show_all is False and
                show_negative_only is False, it shows segments with
                positive contribution to the anomaly score
            number_of_features: example: 6, then it will get the six
                segments with the largest inpat to the anomaly score.
                If None, it will consider all the segments, ignoring
                the segments set by drop_fraction
            drop_fraction: indicates the width of the band to ignore
                segments with explanation weights inside of this band.
                For instance, if it is 0.1, then weights with absolute
                values smaller to 0.1*np.abs(weights).max() would be
                ignore
            save: If True, the image would be save to hard drive
            save_to: path to store the image
            galaxy_name: identification of the spectrum
            save_format: format to store the image
        """
        #######################################################################
        if number_of_features == None:
            # set to the total number of segments
            number_of_features = np.unique(self.segments).size

        #######################################################################
        _, positive_segments = self.get_mask_and_segments(
            positive_only=True,
            number_of_features=number_of_features,
            drop_fraction=drop_fraction,
        )

        #######################################################################
        _, negative_segments = self.get_mask_and_segments(
            positive_only=False,
            negative_only=True,
            number_of_features=number_of_features,
            drop_fraction=drop_fraction,
        )
        #######################################################################
        # to update a plot when in interactive mode with ipython
        plt.clf()

        if show_all is True:

            plt.plot(self.wave, self.galaxy, label="spectrum", color="black")

            plt.plot(
                self.wave, positive_segments, label="positive", color="blue"
            )

            plt.plot(
                self.wave, negative_segments, label="negative", color="red"
            )

        elif show_positive_only is True:

            plt.plot(self.wave, self.galaxy, label="spectrum", color="black")

            plt.plot(
                self.wave, positive_segments, label="positive", color="blue"
            )

        elif show_negative_only:

            plt.plot(self.wave, self.galaxy, label="spectrum", color="black")

            plt.plot(
                self.wave, negative_segments, label="negative", color="red"
            )

        #######################################################################
        plt.legend()

    ###########################################################################
    def get_mask_and_segments(
        self,
        positive_only: bool = True,
        negative_only: bool = False,
        number_of_features: int = 5,
        drop_fraction: float = 0.2,
    ) -> (np.array, np.array):
        #######################################################################
        """
        Get mask and segments according to either a positive or
        negative contribution to the anomaly score.
        Segments obtained with this method will be have NaN values
        where there is no contribution and the actual flux where
        there is a contribution, this to the anomaly score

        INPUT
        positive_only: if True, retrieves segments associated to
            positive weights of the explanation
        negative_only: if positive_only is False and this isTrue,
            retrieves segments associated to negative weights of
            the explanation
        number_of_features: example: 6, then it will get the six
            segments with the largest inpat to the anomaly score.
            If None, it will consider all the segments, ignoring
            the segments set by drop_fraction
        drop_fraction: indicates the width of the band to ignore
            segments with explanation weights inside of this band.
            For instance, if it is 0.1, then weights with absolute
            values smaller to 0.1*np.abs(weights).max() would be
            ignore
        OUTPUT
        (spectrum_mask, explanation_segments)
        spectrum_mask: mask highlighting segments that contribute
            to the anomaly score
        explanation_segments: array with flux values at contributing
            segments, and NaNs otherwise
        """
        #######################################################################
        # check if user inputs right combination of bool values
        bad_input = (positive_only is True) and (negative_only)

        if bad_input is True:

            raise ValueError(
                f"positive_only and negative_only cannot be true"
                f"at the same time."
            )
        #######################################################################
        __, spectrum_mask = self.explanation.get_image_and_mask(
            label=self.explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            num_features=number_of_features,
            # consider all weights by default
            min_weight=-np.inf,
        )
        #######################################################################
        # drop irrelevant explanations
        # segmments in column 0, weights in column 1
        segments_and_weights = np.array(
            self.explanation.local_exp[self.explanation.top_labels[0]]
        )

        # sort weight from segment 0 to the last one
        sort_segments_idx = np.argsort(segments_and_weights[:, 0])
        weights = segments_and_weights[sort_segments_idx, 1]

        # get drop bounds with drop fraction
        # the weights are the same as the values used for heatmap
        heatmap = self.get_heatmap()
        drop = np.abs(heatmap).max() * drop_fraction

        # put NaNs inside the band to ignore
        explanation_segments = np.where(
            np.abs(heatmap) < drop, np.nan, self.galaxy
        )

        # ignore values that do not contribute to the score
        # by adding, I keep track of NaNs in last line of conde
        explanation_segments += np.where(
            spectrum_mask[0, :] == 0, np.nan, self.galaxy
        )

        # *0.5 because of last addition
        return spectrum_mask, explanation_segments * 0.5

    ###########################################################################
    def get_explanation_heatmap(
        self,
        save_map: bool = False,
        symmetric_map: bool = True,
        save_to: str = ".",
        galaxy_name: str = "name",
        save_format: str = "png",
    ) -> (plt.Figure, plt.Axes):
        """
        This method allows to see a color coded spectrum, similar to
        a heatmap, where the color code are the explanation weights

        INPUT
        save_map: If True, the image would be save to hard drive
        symmetric_map: it True, the colorbar will be symmetric
            according to the largest explanation weight in
            absolute value
        save_to: path to store the image
        galaxy_name: identification of the spectrum
        save_format: format to store the image

        OUTPUT
        (fig, ax): of the heatmap
        """
        #######################################################################

        heatmap = self.get_heatmap()

        fig, ax = self._plot_heatmap_spectrum(heatmap, symmetric_map)

        if save_map is True:

            fig.savefig(f"{save_to}/{galaxy_name}_heatmap.{save_format}")

        return fig, ax

    ###########################################################################
    def _plot_heatmap_spectrum(
        self, heatmap: np.array, symmetric_map: bool
    ) -> (plt.Figure, plt.Axes):

        """
        Create a set of line segments so that we can color them
        individually. This creates the points as a N x 1 x 2 array
        so that we can stack points together easily to get the
        segments. The segments array for line collection
        needs to be (numlines) x (points per line) x 2 (for x and y)

        INPUT
        heatmap: array with the explanation weights pixel by pixel
        symmetric_map: if True, colorbar will be symmetric

        OUTPUT
        (fig, ax): of the heatmap
        """

        # Get average since line coloring requires the heatmap
        # size to shrink
        heatmap = 0.5 * (heatmap[:-1] + heatmap[1:])

        points = np.array([self.wave, self.galaxy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))

        # normalize heatmap
        heatmap *= 1 / np.abs(heatmap).max()

        if symmetric_map is True:

            vmin, vmax = -1, 1

        else:
            vmin = heatmap.min()
            vmax = heatmap.max()

        norm = plt.Normalize(vmin, vmax)

        lc = LineCollection(segments, cmap="RdBu", norm=norm)
        lc.set_array(heatmap)
        lc.set_linewidth(1.5)

        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)

        ax.set_xlim(self.wave.min() - 10, self.wave.max() + 10)
        ax.set_ylim(self.galaxy.min() - 1, self.galaxy.max() + 2)

        # plt.show()
        return fig, ax

    ###########################################################################
    def get_heatmap(self) -> np.array:

        """
        Returns array of explanation weights pixel by pixel
        """

        print("Get heat map of explanation", end="\n")

        # there is only one label, since I have a regressor
        ind = self.explanation.top_labels[0]

        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.explanation.local_exp[ind])

        heatmap = np.vectorize(dict_heatmap.get)(self.segments)

        return heatmap