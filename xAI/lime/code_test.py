#! /usr/bin/env python3
import glob
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lib_explanations import Explanation
###################################
def plot(spec, wave_exp, flx_exp, weights_exp, linewidth=0.2,
    cmap='plasma_r', show=False, ipython=False):

    c = weights_exp/np.max(weights_exp)

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(spec, linewidth=linewidth)
    ax.scatter(wave_exp, flx_exp, c=c, cmap=cmap)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

    fig.savefig(f'testing/test.png')
    fig.savefig(f'testing/test.pdf')
    if show:
        plt.show()
    if not ipython:
        plt.close()
######################################################################
ti = time.time()
################################################################################
exp = Explanation()
kernel_widths, data_dict = exp.analyze_explanation(
"results/spec-0308-51662-0081_mad.exp")
spec = np.load("/home/edgar/zorro/SDSSdata/data_proc/spec-0308-51662-0081.npy")
################################################################################
print("###########################################")
for key, val in data_dict.items():
    print(key)
    for idx, kernel in enumerate(val):
        k_with = kernel[0]
        explanation_array = kernel[1]
# print(np.sort(explanation_array[:, 0].astype(np.int)).shape)
################################################################################
wave_exp = explanation_array[:, 0].astype(np.int)
flx_exp = spec[wave_exp]
weights_exp = explanation_array[:, 1]
print(spec.shape, wave_exp.shape)

################################################################################
plot(spec, wave_exp, flx_exp, weights_exp, linewidth=0.2,
    cmap='plasma_r', show=True, ipython=True)
################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
