#! /usr/bin/env python3
import glob
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lib_explanations import Explanation
######################################################################
ti = time.time()
################################################################################
exp = Explanation()
explanation_dict = exp.explanations_from_file(
"results/spec-0308-51662-0081_mad.exp")
spec = np.load("/home/edgar/zorro/SDSSdata/data_proc/spec-0308-51662-0081.npy")
################################################################################
print("###########################################")
for key, value in explanation_dict.items():
    explanation_array = value[1]
    break
################################################################################
wave_exp = explanation_array[:, 0].astype(np.int)
flx_exp = spec[wave_exp]
weights_exp = explanation_array[:, 1]
print(spec.shape, wave_exp.shape)

################################################################################
exp.plot(spec, wave_exp, flx_exp, weights_exp, linewidth=0.2,
    cmap='plasma_r', show=True, ipython=True)
################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
