"""Functionality to explain regressors trained on spectra"""
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb

from tqdm.auto import tqdm
from lime import lime_base
from lime.lime_image import ImageExplanation
from astroExplain.spectra.fudge import Fudge

###############################################################################
class LimeSpectraExplainer:
    """
    Taken from lime original repository and adapted to explain a regressor
    model for spectra: https://github.com/marcotcr/lime

    Explains predictions on Spectra (i.e. flat matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data.
    """

    def __init__(
        self,
        kernel_width=0.25,
        kernel=None,
        verbose=False,
        feature_selection="auto",
        random_state=None,
    ):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1).
                If None, defaults to an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel_fn(d, kernel_width):

                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

            kernel_fn = partial(kernel_fn, kernel_width=kernel_width)

        else:
            kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )

        # compatibility

    def explain_instance(
        self,
        spectrum,
        classifier_fn,
        segmentation_fn,
        fudge_parameters: dict,
        explainer_parameters: dict,
        # num_features=100000,
        # num_samples=1000,
        # batch_size=10,
        # distance_metric="cosine",
        # model_regressor=None,
        # random_seed=None,
        # progress_bar=True,
        # see hard code of values below
        # labels=(1,),
        # top_labels=5,
    ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
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
            amplitude: amplitude of gaussians or white-noise
            mu: mean of white-noise
            std: standard deviation of gaussians or white-noise
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.

        Returns:
            An ImageExplanation object (see lime_image.py) with the
            corresponding explanations.
        """

        # set variables bellow to accomadate for a regressor
        labels = None
        top_labels = 1
        # not in use, so removed from function call
        model_regressor = None

        # convert to rgb image
        if len(spectrum.shape) == 2:
            spectrum = gray2rgb(spectrum)

        segments = segmentation_fn(spectrum)

        fudged_spectrum = self.fudge_spectrum(
            spectrum[0, :, 0], segments, fudge_parameters
        )
        fudged_spectrum = gray2rgb(fudged_spectrum.reshape(1, -1))

        top = labels

        data, labels = self.data_labels(
            spectrum,
            fudged_spectrum,
            segments,
            classifier_fn,
            explainer_parameters["number_samples"],
            batch_size=explainer_parameters["batch_size"],
            progress_bar=explainer_parameters["progress_bar"],
        )
        #######################################################################

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=explainer_parameters["distance_metric"],
        ).ravel()

        ret_exp = ImageExplanation(spectrum, segments)

        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()

        for label in top:
            (
                ret_exp.intercept[label],
                ret_exp.local_exp[label],
                ret_exp.score[label],
                ret_exp.local_pred[label],
            ) = self.base.explain_instance_with_data(
                data,
                labels,
                distances,
                label,
                explainer_parameters["number_features"],
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )

        return ret_exp

    ###########################################################################
    def data_labels(
        self,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_samples,
        batch_size=10,
        progress_bar=True,
    ):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(
            0, 2, num_samples * n_features
        ).reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        # rows = data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

    ###########################################################################
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
                sigma=fudge_parameters["sigma"],
            )

        else:
            raise NotImplementedError(f"{kind_of_fudge} fudge not implemented")

        return fudged_spectrum
