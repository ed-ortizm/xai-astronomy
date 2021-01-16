import csv
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model
###############################################################################
# def top_reconstructions(scores, n_normal_outliers=30):
#     """Selecting top outliers for a given outlier score and its SDSS metadata"""
#
#     spec_idxs = np.argpartition(scores,
#     [n_normal_outliers, -1*n_normal_outliers])
#
#     most_normal = spec_idxs[: n_normal_outliers]
#     most_oulying = spec_idxs[-1*n_normal_outliers:]
#
#     return most_normal, most_oulying
###############################################################################
AE_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'

# def mse_score(O, model_path=AE_path):
#
#     """
#     Using my UL ODA together with the outlier score to transfor the model
#     into a regression model to feed it to the LIME explanations
#     f: outlier_score(O, AE.predict(O))
#     """
#
#     model_name = model_path.split('/')[-1]
#     print(f'Loading model: {model_name}')
#     AE = load_model(f'{model_path}')
#
#     if O.shape[0] == 3801:
#         O = O.reshape(1,-1)
#
#     print(f'Computing the predictions of {model_name}')
#     R = AE.predict(O)
#
#     return np.square(R-O).mean(axis=1)
###############################################################################
class Explainer:
    def __init__(self, training_data, training_labels, feature_names,
    kernel_width, feature_selection, training_data_stats=None,
    sample_around_instance=False, discretize_continuous=False,
    discretizer='decile',verbose=True,mode='regression'):

        self.tr_data = training_data
        slef.tr_labels = training_labels
        self.ftr_names = feature_names
        self.k_width = kernel_width
        self.ftr_selct = feature_selection
        self.tr_data_stats = training_data_stats
        self.sar_instance = sample_around_instance
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

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

    def explanation(self, sdss_name='sdss_name', html=False, figure=False):

        xpl = self.explainer.explain_instance(outlier, mse_score,
              num_features=num_features)

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

    def analyze_explanation(self, spec_path, exp_file_path):

        if os.path.exists(spec_path):
            spec = np.load(spec_path)
        else:
            print(f'There is no file {spec_path}')
            return None

        if os.path.exists(exp_file_path):
            exp = self.process_explanation(exp_file_path)
        else:
            print(f'There is no file {exp_file_path}')
            return None


        wave_exp = exp[:, 0].astype(np.int)
        flx_exp = spec[wave_exp]
        weights_exp = exp[:, 1]

        return wave_exp, flx_exp, weights_exp

    def plot_explanation(self, spec_path, exp_file_path, linewidth=0.2, cmap='plasma_r'):

        if os.path.exists(spec_path):
            spec = np.load(spec_path)
        else:
            print(f'There is no file {spec_path}')
            return None

        wave_exp, flx_exp, weights_exp = self.analyze_explanation(spec_path,
        exp_file_path)

        c = weights_exp/np.max(weights_exp)

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, c=c, cmap=cmap)

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

        fig.savefig(f'test.png')
        fig.savefig(f'test.pdf')

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
# scatter lime weights vs mse outlier score
