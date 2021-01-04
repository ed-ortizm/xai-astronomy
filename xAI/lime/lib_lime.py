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

class explanations:

    def __init__(self, discretize_continuous=False):
        self.discretize_continuous = discretize_continuous

    def process_explanations(self, exp_file_path):

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
        explanation = list(dict.fromkeys(exp))
        explanation.remove(',')
        explanation.remove('')
        explanation.remove('\r\n')

        n_features = len(explanation)
        n_values = len(explanation[0].split(','))

        feature_weight = np.empty((n_features, n_values))


        if not discretize_continuous:

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
################################################################################

###############################################################################
