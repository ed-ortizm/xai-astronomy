import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import lime
import lime.lime_tabular

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from tensorflow.keras.models import load_model
###############################################################################
class Explainer:
    def __init__(self, explainer_type, training_data, training_labels,
        feature_names, kernel_width, feature_selection,
        training_data_stats=None, sample_around_instance=False,
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

    def explanation(self, x, regressor, sdss_name='sdss_name', html=False,
    figure=False):

        xpl = self.explainer.explain_instance(x, regressor,
            num_features=x.shape[0])

        if html:
            xpl.save_to_file(file_path = f'{html_name}.html')

        if figure:
            fig = xpl.as_pyplot_figure()
            fig.savefig(f'{sdss_name}.pdf')

        return xpl.as_list()

class Explanation:

    def __init__(self, discretize_continuous=False):
        self.discretize_continuous = discretize_continuous

    def process_explanation(self, exp_file_path):

        if os.path.exists(exp_file_path):
            # Extracting kernel width
            k_width = exp_file_path.split('/')[-1].split('_')[0]
        else:
            print(f'There is no file {exp_file_path}')
            return None

        explanation = None

        with open(f'{exp_file_path}', newline='\n') as file:

            for line in file:
                explanation = line

        explanation = explanation.split('"')
        explanation = list(dict.fromkeys(explanation))
        explanation.remove(',')
        explanation.remove('')
        explanation.remove('\r\n')

        n_features = len(explanation)
        n_values = len(explanation[0].split(','))

        feature_weight = np.empty((n_features, n_values))


        if not self.discretize_continuous:

            for feature_idx, tuple in enumerate(explanation):

                tuple = tuple.split(',')

                tuple[0] = np.float(tuple[0].strip("('flux")) - 1.0
                tuple[1] = np.float(tuple[1].strip(')'))

                feature_weight[feature_idx, :] = np.array(tuple)

        else:

            tuple = tuple.split(',')

            tuple[0] = tuple[0][2:-1]
            tuple[1] = np.float(tuple[1][:-1])

            if '<' in tuple[0]:

                if len(tuple[0].split('<'))==2:
                    tuple[0] = np.int(tuple[0].split('<')[0])
                else:
                    tuple[0] = np.int(tuple[0].split('<')[1])

            else:

                tuple[0] = np.int(tuple[0].split('>')[0])

            feature_weight[feature_idx, :] = np.array(tuple)

        print(f'numpy array created: [feature, lime_weight]')

        return feature_weight

    def analyze_explanation(self, x, exp_file_path):

        if os.path.exists(exp_file_path):
            exp = self.process_explanation(exp_file_path)
        else:
            print(f'There is no file {exp_file_path}')
            return None

        wave_exp = exp[:, 0].astype(np.int)
        flx_exp = x[wave_exp]
        weights_exp = exp[:, 1]

        return wave_exp, flx_exp, weights_exp

    def plot(self, spec, wave_exp, flx_exp, weights_exp, linewidth=0.2,
        cmap='plasma_r', show=False):

        c = weights_exp/np.max(weights_exp)

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, c=c, cmap=cmap)

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

        fig.savefig(f'test.png')
        fig.savefig(f'test.pdf')
        if show:
            plt.show()
        plt.close()

class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    def __init__(self, model_path, o_scores_path='.', metric='mse', p='p',
        n_spec=30, custom=False, custom_metric=None):
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

            n_spec: (int > 0) this parameter contros the number of objects identifiers to
                return for the top reconstruction, that is the most oulying and
                the most normal objects
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

        if O.shape[0] == 3801:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return O, R

    def score(self, O):
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
#  Explore this idea: like the setters and getters for the custom fucntion ;)
# class Helper(object):
#
#     def add(self, a, b):
#         return a + b
#
#     def mul(self, a, b):
#         return a * b
#
#
# class MyClass(Helper):
#
#     def __init__(self):
#         Helper.__init__(self)
#         print self.add(1, 1)
#
#
# if __name__ == '__main__':
#     obj = MyClass()

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
                object in the training set.
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
################################################################################
def segment_spec(spec, n_segments, training_data_path):

    segments = ['median', 'gray?', 'average', 'flat', 'other']
    pass
###############################################################################
## Case for spectra
class Spec_segmenter:
    def __init__(self):
        pass
# segments_slic = slic(spec_test, n_segments=30, compactness=100,
#     sigma=1, multichannel=False)
#
# boundaries = mark_boundaries(spec_test, segments_slic)
#
# diff = boundaries[0, :, 0] - spec_test[0, :]
#
# idxs = np.nonzero(diff)[0]
#
# plot(spec_test[0, :], color='black', linewidth=0.7)
#
# for vline in idxs:
#     axvline(x=vline, color='red', linewidth=0.5)

###############################################################################
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
