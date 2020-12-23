import csv
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
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

def mse_score(O):

    # Loading trained model
    model_path = '/home/edgar/zorro/outlier_AEs/trained_models/AutoEncoder'
    AE = load_model(f'{model_path}')

    if O.shape[0] == 3801:
        O = O.reshape(1,-1)
    R = AE.predict(O)

    return np.square(R-O).mean(axis=1)

def explain(kernel_width, training_data, training_labels, data_row,
                predict_fn, num_features, file):

    print(f'Explainer with kernel width: {kernel_width}')

    explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=training_data, mode='regression',
    raining_labels = training_labels, kernel_width=kernel_width, verbose=True)

    print(f'Computing explanations for the top MSE outlier')
    exp = explainer.explain_instance(data_row=data_row, predict_fn=predict_fn,
    num_features=num_features)

    print(f'Saving explanation as html')
    exp.save_to_file(file_path=f'./kw_{kernel_width}_explanation_AE_p.html')

    # explanation as list
    print(f'Explanation list to a file...')
    exp_list = exp.as_list()
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(exp_list)

    # explanation as pyplot figure
    print(f'Explanation to pyplot figure')
    exp_fig = exp.as_pyplot_figure()
    exp_fig.savefig(f'./kw_{kernel_width}_explanation_AE_p.png')

    return exp
