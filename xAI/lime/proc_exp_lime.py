#! /usr/bin/env python3

import time

import matplotlib.pyplot as plt
import numpy as np

ti = time.time()
################################################################################
k_widths = [5, 10, 20, 38, 50, 75, 100, 250, 500, 1_000]
explanations = {}

with open('explanations_list.csv', newline='\n') as file:

    for idx, line in enumerate(file):
        explanations[f'kernel width = {k_widths[idx]}'] = line

for kernel_key in explanations:

    exp = explanations[kernel_key]
    exp = exp.split('"')
    exp = list(dict.fromkeys(exp))
    exp.remove(',')
    exp.remove('')
    exp.remove('\r\n')

    explanations[kernel_key] = exp

feature_weight = np.empty((10, 100, 3)) # 10 kernels, 100 explained features,
# feature and value --> (10, 100, 2)

for kernel_idx, kernel_key in enumerate(explanations):

    exp: explanations[kernel_key]
    for feature_idx, tuple in enumerate(exp):

        tuple = tuple.split(',')

        tuple[0] = tuple[0][2:-1]
        tuple[1] = np.float(tuple[1][:-1])

        if '<' in tuple[0]:

            if len(tuple[0].split('<'))==2:
                tuple[0] = np.int(tuple[0].split('<')[0])
            else:
                tuple[0] = np.int(tuple[0].split('<')[1])

        else:
            # print(tuple[0].split('>')[0], tuple[0].split('>')[1])
            tuple[0] = np.int(tuple[0].split('>')[0])

        tuple.append(k_widths[kernel_idx])
        feature_weight[kernel_idx, feature_idx, :] = np.array(tuple)

print(feature_weight[0, 0, :])
np.save('features_exp_weights_10_kwidths.npy', feature_weight)

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
