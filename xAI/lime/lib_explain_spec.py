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

    def __init__(self, model_path, o_scores_path, metric='mse', p='p', n_spec=30):

        self.model_path = model_path
        self.o_scores_path = o_scores_path
        self.metric = metric
        self.p = p
        self.n_spec = n_spec

    def score(self, O):

        model_name = self.model_path.split('/')[-1]
        print(f'Loading model: {model_name}')
        model = load_model(f'{self.model_path}')

        if self.metric == 'mse':
            print(f'Computing the predictions of {model_name}')
            return self._mse(O=O, model=model)

        elif self.metric == 'chi2':
            print(f'Computing the predictions of {model_name}')
            return self._chi2(O=O, model=model)

        elif self.metric == 'mad':
            print(f'Computing the predictions of {model_name}')
            return self._mad(O=O, model=model)

        elif self.metric == 'lp':

            if self.p == 'p' or self.p <= 0:
                print(f'For the {self.metric} metric you need p')
                return None

            print(f'Computing the predictions of {model_name}')
            return self._lp(O=O, model=model)

        else:
            print(f'The provided metric: {self.metric} is not implemented yet')
            return None

    def _mse(self, O, model):

        if O.shape[0] == 3801:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return np.square(R-O).mean(axis=1)

    def _chi2(self, O, model):

        if O.shape[0] == 3801:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return (np.square(R-O)*(1/np.abs(R))).mean(axis=1)

    def _mad(self, O, model):

        if O.shape[0] == 3801:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return np.abs(R-O).mean(axis=1)

    def _lp(self, O, model):

        if O.shape[0] == 3801:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return (np.sum((np.abs(R-O))**self.p, axis=1))**(1/self.p)

    def metadata(self, spec_idx, training_data_files):

        # print('Gathering name of data points used for training')

        sdss_names = [name.split('/')[-1].split('.')[0] for name in
            training_data_files]

        # print('Retrieving the sdss name of the desired spectrum')

        sdss_name = sdss_names[spec_idx]
        sdss_name_path = training_data_files[spec_idx]

        return sdss_name, sdss_name_path

    def top_reconstructions(self, O):

        """
        Selecting top outliers for a given outlier score and its SDSS metadata
        """
        if os.path.exists(f"{self.o_scores_path}/{self.metric}_o_score.npy"):
            scores= np.load(f"{self.o_scores_path}/{self.metric}_o_score.npy")
        else:
            scores = self.score(O)

        spec_idxs = np.argpartition(scores,
        [self.n_spec, -1*self.n_spec])

        most_normal = spec_idxs[: self.n_spec]
        most_oulying = spec_idxs[-1*self.n_spec:]

        ## Retrieving metadata
        # later

        return most_normal, most_oulying



################################################################################
def segment_spec(spec, n_segments, training_data_path):

    segments = ['median', 'gray?', 'average', 'flat']
    pass
###############################################################################
# scatter lime weights vs mse outlier score