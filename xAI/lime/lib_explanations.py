from functools import partial
import glob
from itertools import product
import os
import sys

import matplotlib
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import lime
# import lime.lime_tabular
from lime import lime_tabular
from lime import lime_image
import multiprocessing as mp
import pickle
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from tensorflow.keras.models import load_model
###############################################################################
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
class PlotData:

    def __init__(self, spec, sdss_name, vmin, vmax):
        self.spec = spec
        self.sdss_name = sdss_name
        self.vmin = vmin
        self.vmax = vmax
        self._fig = None
        self._cmap = None

    def _colorbar_explanation(self):
        # Make axes with dimensions as desired.
        ax_cb = self._fig.add_axes([0.91, 0.05, 0.03, 0.9])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        self._cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=self._cmap,
                                        norm=norm,
                                        orientation='vertical', extend='both')
        cb.set_label('Normalized weights')


        return cb

    def plot_explanation(self,
        wave_exp, flx_exp, weights_explanation,
        kernel_width, feature_selection, metric,
        s=3., linewidth=1., alpha=0.7):

        a = np.sort(weights_explanation)
        print([f'{i:.2E}' for i in a[:2]])
        print([f'{i:.2E}' for i in a[-2:]])

        self._fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(left=0.08, right=0.9)

        line, = ax.plot(self.spec, linewidth=linewidth, alpha=alpha)

        scatter = ax.scatter(wave_exp, flx_exp, s=s,
            c=weights_explanation, cmap='plasma',
            vmin=self.vmin, vmax=self.vmax, alpha=1.)

        ax_cb = self._colorbar_explanation()
        ax.set_title(
        f'{self.sdss_name}: {metric}, {feature_selection}, k_width={kernel_width}')

        # plt.tight_layout()

        return self._fig, ax, ax_cb, line, scatter
###############################################################################
class ExplanationData:

    def __init__(self, explanation_file):

        self.explanation_file = explanation_file
        self.sdss_directory = "/home/edgar/Documents/pyhacks/interactive_plotting"
        # "/home/edgar/zorro/SDSSdata/data_proc"
        self.sdss_name = self.explanation_file.split('_')[0]
        # self.explanation_file.split('/')[-1].split('_')[0]
        self.spec = np.load(f'{self.sdss_directory}/{self.sdss_name}.npy')

    def get_explanation_data(self, n_line):

        explanation_dictionary = self.get_serialized_data()

        kernel_width = explanation_dictionary[f'{n_line}'][0]
        kernel_width = float(kernel_width)

        array_explanation = explanation_dictionary[f'{n_line}'][1]
        wave_explanation = array_explanation[:, 0].astype(np.int)
        flux_explanation = self.spec[wave_explanation]
        weights_explanation = array_explanation[:, 1]
        metric = explanation_dictionary[f'{n_line}'][3]
        feature_selection = explanation_dictionary[f'{n_line}'][4]

        return (wave_explanation,
                flux_explanation,
                weights_explanation,
                kernel_width, metric, feature_selection)

    def get_serialized_data(self):

         with open(f'{self.explanation_file}', 'rb') as file:
             return pickle.load(file)
###############################################################################
class Explainer_parallel:

    def __init__(self, explainer_type, training_data, training_labels,
        feature_names, kernel_widths, features_selection,
        sample_around_instance, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=False,
        mode='regression', n_processes=None):
        # The input variable are lists

        self.k_widths = kernel_widths
        self.ftrs_slect = features_selection
        self.around_instance = sample_around_instance

        if n_processes == None:
            self.n_processes = mp.cpu_count()-1 or 1
        else:
            self.n_processes = n_processes


        # Fixed values

        self.xpl_type = explainer_type
        self.t_dat = training_data
        self.t_lbls = training_labels
        self.ftr_names = feature_names
        self.t_dat_stats = training_data_stats
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

        self.Ex_partial = partial(Explainer, explainer_type=self.xpl_type,
        training_data=self.t_dat, training_labels=self.t_lbls,
        feature_names=self.ftr_names, training_data_stats=self.t_dat_stats,
        discretize_continuous=self.discretize, discretizer=self.discretizer,
        verbose=self.verbose, mode=self.mode)

        self.explanations = None

    def get_explanations(self, x, regressor, sdss_name):

        params_grid = product([x], [regressor], [sdss_name],
            self.k_widths, self.ftrs_slect, self.around_instance)

        with mp.Pool(processes=self.n_processes) as pool:
            self.explanations = pool.starmap(self._get_explanation, params_grid)
            self._sizeof(self.explanations, itr_name='explanations')

        return self.explanations

    def _get_explanation(self, x, regressor, sdss_name,
        kernel_width, feature_selection, sample_around_instance):

        explainer = self.Ex_partial(kernel_width, feature_selection,
            sample_around_instance)

        self._sizeof(explainer, itr_name='explainer', is_itr=False)

        return [sdss_name, kernel_width, feature_selection, sample_around_instance,
            explainer.explanation(x, regressor)]

    def _sizeof(self, iterable, itr_name="iterable", is_itr=True):

        if is_itr:
            size = 0
            for itr in iterable:
                x = sys.getsizeof(itr)*1e-6
                print(f'The size of object from {itr_name} is: {x:.2f} Mbs')
                size += x
            print(f"The total size of {itr_name} is {size:.2f} Mbs")
        else:
            size =  sys.getsizeof(iterable)
            print(f"The total size of {itr_name} is {size:.2f} Mbs")
###############################################################################
class Explainer:
    def __init__(self, kernel_width, feature_selection,
        sample_around_instance, explainer_type, training_data,
        training_labels, feature_names, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=True,
        mode='regression'):

        self.xpl_type = explainer_type
        self.tr_data = training_data
        self.tr_labels = training_labels
        self.ftr_names = feature_names
        self.k_width = kernel_width
        self.ftr_select = feature_selection
        self.tr_data_stats = training_data_stats
        self.sar_instance = sample_around_instance
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

        if self.xpl_type == "tabular":

            self.explainer = self._tabular_explainer()
            x = sys.getsizeof(self.explainer)*1e-6
            print(f'The size of the explainer is: {x:.2f} Mbs')

    def _tabular_explainer(self):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.tr_data, training_labels=self.tr_labels,
            feature_names=self.ftr_names, kernel_width=self.k_width,
            feature_selection=self.ftr_select,
            training_data_stats=self.tr_data_stats,
            sample_around_instance=self.sar_instance,
            discretize_continuous=self.discretize, discretizer=self.discretizer,
            verbose = self.verbose, mode=self.mode)

        return explainer

    def explanation(self, x, regressor):


        xpl = self.explainer.explain_instance(x, regressor,
            num_features=x.shape[0])
        return xpl.as_list()
###############################################################################
class Explanation:

    def __init__(self, discretize_continuous=False):
        self.discretize_continuous = discretize_continuous

    def explanations_from_file(self, explanation_file_path: str, save=True):

        if not os.path.exists(explanation_file_path):

            print(f'There is no file {explanation_file_path}')
            return None

        sdss_name = explanation_file_path.split("/")[-1].split("_")[0]
        metric = explanation_file_path.split("/")[-1].split("_")[1].strip(".exp")
        explanation_dict = {}

        with open(explanation_file_path, "r") as file:

            explanation_lines = file.readlines()

            for idx, explanation_line in enumerate(explanation_lines):

                explanation_line = self._line_curation(explanation_line)

                k_width = explanation_line[1] # string
                feature_selection = explanation_line[2]
                sample_around_instance = explanation_line[3]

                explanation_array = self._fluxes_weights(
                    line=explanation_line[4:])

                explanation_dict[f'{idx}'] = [k_width, explanation_array,
                    sdss_name, metric,
                    f'{feature_selection}_{sample_around_instance}']

        return explanation_dict

    def _fluxes_weights(self, line):

        length = np.int(len(line)/2)
        fluxes_weights = np.empty((length,2))

        for idx, fw in enumerate(fluxes_weights):
            fw[0] = np.float(line[2*idx].strip("'flux "))
            fw[1] = np.float(line[2*idx+1])

        return fluxes_weights

    def _line_curation(self, line):
        for charater in "()[]'":
            line = line.replace(charater, "")
        return [element.strip(" \n") for element in line.split(",")]

    def plot_explanation(self, spec, wave_exp, flx_exp, weights_exp, s=10., linewidth=0.2, cmap='plasma_r', show=False, ipython=False):

        c = weights_exp/np.max(weights_exp)

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)

        fig.colorbar()

        fig.savefig(f'testing/test.png')
        fig.savefig(f'testing/test.pdf')
        if show:
            plt.show()
        if not ipython:
            plt.close()
###############################################################################
class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    def __init__(self, model_path, o_scores_path='.', metric='mse', p='p',
        n_spec=100, custom=False, custom_metric=None):
        """
        Init fucntion

        Args:
            model_path: path where the trained generative model is located

            o_scores_path: (str) path where the numpy arrays with the outlier scores
                is located. Its functionality is for the cases where the class
                will be implemented several times, offering therefore the
                possibility to load the scores instead of computing them over
                and over.

            metric: (str) the name of the metric used to compute the outlier score
                using the observed spectrum and its reconstruction. Possible

            p: (float > 0) in case the metric is the lp metric, p needs to be a non null
                possitive float [Aggarwal 2001]

            n_spec: (int > 0) this parameter controls the number of objects identifiers to
                return for the top reconstruction, that is the most oulying and
                the most normal objects --> move it to the method
        """

        self.model_path = model_path
        self.o_scores_path = o_scores_path
        self.metric = metric
        self.p = p
        self.n_spec = n_spec
        self.custom = custom
        if self.custom:
            self.custom_metric = custom_metric

    def _get_OR(self, O, model):

        if len(O.shape) == 1:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return O, R

    def score(self, O):
        print("Score working in parallelization")
        """
        Computes the outlier score according to the metric used to define
        instantiate the class.

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

        Returns:
            A one dimensional numpy array with the outlier scores for objects
            present in O
        """

        model_name = self.model_path.split('/')[-1]
        print(f'Loading model: {model_name}')
        model = load_model(f'{self.model_path}')

        O, R = self._get_OR(O, model)

        if self.custom:
            print(f'Computing the predictions of {model_name}')
            return self.user_metric(O=O, R=R)

        elif self.metric == 'mse':
            print(f'Computing the predictions of {model_name}')
            return self._mse(O=O, R=R)

        elif self.metric == 'chi2':
            print(f'Computing the predictions of {model_name}')
            return self._chi2(O=O, R=R)

        elif self.metric == 'mad':
            print(f'Computing the predictions of {model_name}')
            return self._mad(O=O, R=R)

        elif self.metric == 'lp':

            if self.p == 'p' or self.p <= 0:
                print(f'For the {self.metric} metric you need p')
                return None

            print(f'Computing the predictions of {model_name}')
            return self._lp(O=O, R=R)

        else:
            print(f'The provided metric: {self.metric} is not implemented yet')
            return None

    def _coscine_similarity(self, O, R):
        """
        Computes the coscine similarity between the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the cosine similarity between
            objects O and their reconstructiob
        """

        pass

    def _jaccard_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        pass

    def _sorensen_dice_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """
        pass
# Mahalanobis, Canberra, Braycurtis, and KL-divergence
    def _mse(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        return np.square(R-O).mean(axis=1)

    def _chi2(self, O, R):
        """
        Computes the chi square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the chi square error for objects
            present in O
        """

        return (np.square(R-O)*(1/np.abs(R))).mean(axis=1)

    def _mad(self, O, R):
        """
        Computes the maximum absolute deviation from the reconstruction of the
        input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the maximum absolute deviation
            from the objects present in O
        """

        return np.abs(R-O).mean(axis=1)

    def _lp(self, O, R):
        """
        Computes the lp distance from the reconstruction of the input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the lp distance from the objects
            present in O
        """

        return (np.sum((np.abs(R-O))**self.p, axis=1))**(1/self.p)
# gotta code conditionals to make sure that the user inputs a "good one"
    def user_metric(self, custom_metric, O, R):
        """
        Computes the custom metric for the reconstruction of the input objects
        as defined by the user

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the score produced by the user
            defiend metric of objects present in O
        """

        return self.custom_metric(O, R)

    def metadata(self, spec_idx, training_data_files):
        """
        Generates the names and paths of the individual objects used to create
        the training data set.
        Note: this work according to the way the training data set was created

        Args:
            spec_idx: (int > 0) the location index of the spectrum in the
                training data set.

            training_data_files: (list of strs) a list with the paths of the
                individual objects used to create the training data set.

        Returns:
            sdss_name, sdss_name_path: (str, str) the sdss name of the objec,
                the path of the object in the files system
        """


        # print('Gathering name of data points used for training')

        sdss_names = [name.split('/')[-1].split('.')[0] for name in
            training_data_files]

        # print('Retrieving the sdss name of the desired spectrum')

        sdss_name = sdss_names[spec_idx]
        sdss_name_path = training_data_files[spec_idx]

        return sdss_name, sdss_name_path

    def top_reconstructions(self, O):
        """
        Selects the most normal and outlying objecs

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

        Returns:
            most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
                with the location indexes of the most normal and outlying
                object in the training (and pred) set.
        """

        if os.path.exists(f"{self.o_scores_path}/{self.metric}_o_score.npy"):
            scores= np.load(f"{self.o_scores_path}/{self.metric}_o_score.npy")
        else:
            scores = self.score(O)

        spec_idxs = np.argpartition(scores,
        [self.n_spec, -1*self.n_spec])

        most_normal = spec_idxs[: self.n_spec]
        most_oulying = spec_idxs[-1*self.n_spec:]

        return most_normal, most_oulying
###############################################################################
class ImageExplainer:
    def __init__(self):
        pass
###############################################################################
## Case for spectra
class Spec_segmenter:

    def __init__(self, spec, segmenter="slic"):

        if len(spec.shape) == 1:
            self.spec = spec.reshape(1,-1)
        else:
            self.spec = spec

        self.segmenter = segmenter

    def slic(self, n_segments=30, compactness=100, sigma=1):

        segments = slic(self.spec, n_segments=n_segments,
            compactness=compactness, sigma=sigma, multichannel=False)

        boundaries = mark_boundaries(self.spec, segments)

        diff = boundaries[0, :, 0] - self.spec[0, :]

        idxs = np.nonzero(diff)[0]

        return idxs


    def plot(self, idxs, show=False):

        fig, ax = plt.subplots(figsize=(15, 5))

        wave = np.arange(self.spec.shape[1])

        for n, idx in enumerate(idxs):

            if n == 0:
                ax.plot(wave[:idxs[n]], self.spec[0, :idxs[n]], linewidth=0.7)
            else:
                ax.plot(wave[idxs[n-1]: idxs[n]],
                    self.spec[0, idxs[n-1]: idxs[n]], linewidth=0.7)

        if show:
            plt.show()

    ###########################################################################
    # print(f"Felzenszwalb's efficient graph based segmentation")
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    #
    # print(f'Quickshift image segmentation')
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
    #
    # print(f'Compact watershed segmentation of gradient images')
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    # print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
    """
    ====================================================
    Comparison of segmentation and superpixel algorithms
    ====================================================

    This example compares four popular low-level image segmentation methods.  As
    it is difficult to obtain good segmentations, and the definition of "good"
    often depends on the application, these methods are usually used for obtaining
    an oversegmentation, also known as superpixels. These superpixels then serve as
    a basis for more sophisticated algorithms such as conditional random fields
    (CRF).


    Felzenszwalb's efficient graph based segmentation
    -------------------------------------------------
    This fast 2D image segmentation algorithm, proposed in [1]_ is popular in the
    computer vision community.
    The algorithm has a single ``scale`` parameter that influences the segment
    size. The actual size and number of segments can vary greatly, depending on
    local contrast.

    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004


    Quickshift image segmentation
    -----------------------------

    Quickshift is a relatively recent 2D image segmentation algorithm, based on an
    approximation of kernelized mean-shift. Therefore it belongs to the family of
    local mode-seeking algorithms and is applied to the 5D space consisting of
    color information and image location [2]_.

    One of the benefits of quickshift is that it actually computes a
    hierarchical segmentation on multiple scales simultaneously.

    Quickshift has two main parameters: ``sigma`` controls the scale of the local
    density approximation, ``max_dist`` selects a level in the hierarchical
    segmentation that is produced. There is also a trade-off between distance in
    color-space and distance in image-space, given by ``ratio``.

    .. [2] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008


    SLIC - K-Means based image segmentation
    ---------------------------------------

    This algorithm simply performs K-means in the 5d space of color information and
    image location and is therefore closely related to quickshift. As the
    clustering method is simpler, it is very efficient. It is essential for this
    algorithm to work in Lab color space to obtain good results.  The algorithm
    quickly gained momentum and is now widely used. See [3]_ for details.  The
    ``compactness`` parameter trades off color-similarity and proximity, as in the
    case of Quickshift, while ``n_segments`` chooses the number of centers for
    kmeans.

    .. [3] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Suesstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.


    Compact watershed segmentation of gradient images
    -------------------------------------------------

    Instead of taking a color image as input, watershed requires a grayscale
    *gradient* image, where bright pixels denote a boundary between regions.
    The algorithm views the image as a landscape, with bright pixels forming high
    peaks. This landscape is then flooded from the given *markers*, until separate
    flood basins meet at the peaks. Each distinct basin then forms a different
    image segment. [4]_

    As with SLIC, there is an additional *compactness* argument that makes it
    harder for markers to flood faraway pixels. This makes the watershed regions
    more regularly shaped. [5]_

    .. [4] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29

    .. [5] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf
    """
###############################################################################
