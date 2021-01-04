import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model
###############################################################################
def top_reconstructions(scores, n_normal_outliers=20):
    """Selecting top outliers for a given outlier score and its SDSS metadata"""

    spec_idxs = np.argpartition(scores,
    [n_normal_outliers, -1*n_normal_outliers])

    most_normal = spec_idxs[: n_normal_outliers]
    most_oulying = spec_idxs[-1*n_normal_outliers:]

    return most_normal, most_oulying
###############################################################################
AE_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'

def mse_score(O, model_path=AE_path):

    """
    Using my UL ODA together with the outlier score to transfor the model
    into a regression model to feed it to the LIME explanations
    f: outlier_score(O, AE.predict(O))
    """

    model_name = model_path.split('/')[-1]
    print(f'Loading model: {model_name}')
    AE = load_model(f'{model_path}')

    if O.shape[0] == 3801:
        O = O.reshape(1,-1)

    print(f'Computing the predictions of {model_name}')
    R = AE.predict(O)

    return np.square(R-O).mean(axis=1)
###############################################################################

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

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, c=c, cmap=cmap)

        fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

        fig.savefig(f'test.png')
        fig.savefig(f'test.pdf')

        plt.close()

################################################################################
def segment_spec(spec, n_segments=100):

    return
###############################################################################
