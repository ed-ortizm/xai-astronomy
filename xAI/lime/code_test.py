#! /usr/bin/env python3
import glob
import os
import sys
import time

import numpy as np

from lib_explanations import Explanation

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
    # print(len(val))
    for idx, kernel in enumerate(val):
        # print(idx+1, kernel[0], kernel[1].shape)
        k_with = kernel[0]
        explanation_array = kernel[1]
print(np.sort(explanation_array[:, 0].astype(np.int)).shape)
################################################################################
wave_exp = explanation_array[:, 0].astype(np.int)
flx_exp = spec[wave_exp]
weights_exp = explanation_array[:, 1]
################################################################################
linewidth=0.2
cmap='plasma_r'
show=False
ipython=False

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



# idxx = []
# for idx, weights in enumerate(weights_exp):
#     max_val = np.max(weights)
#     if max_val != 0.0 and not np.isnan(max_val):
#         idxx.append(idx)
# ################################################################################
# wwave, fflx, cc = [], [], []
# for idx in idxx[:1]:
#     wave, flx, c = exp_plot(idx=idx, spec=spec, wave_exp=wave_exp,
#         flx_exp=flx_exp, weights_exp=weights_exp, linewidth=0.5,
#         cmap='plasma_r', s=5, show=True)
#     wwave.append(wave)
#     fflx.append(flx)
#     cc.append(c)
# array_exp = np.load(
# "testing/spec-0308-51662-0081_metric_mad_lasso_path_around_False.npy")
# wave_exp, flx_exp, weights_exp = exp.process_array(spec, array_exp)

################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.collections as mcoll
#
# def multicolored_lines(idx, x, y, cmap="hsv"):
#     """
#     http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#     http://matplotlib.org/examples/pylab_examples/multicolored_line.html
#     """
#
#     sort_x = np.argsort(x[idx])
#     x = x[idx][sort_x]
#     y = y[idx][sort_x]
#     fig, ax = plt.subplots()
#     lc = colorline(x, y, cmap=cmap)
#     plt.colorbar(lc)
#     plt.xlim(x.min() - 10, x.max() + 10)
#     plt.ylim(-1., 1.1*nanmax(y))
#     plt.show()
#
# def colorline(
#         x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
#         linewidth=3, alpha=1.0):
#     """
#     http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
#     http://matplotlib.org/examples/pylab_examples/multicolored_line.html
#     Plot a colored line with coordinates x and y
#     Optionally specify colors in the array z
#     Optionally specify a colormap, a norm function and a line width
#     """
#
#     # Default colors equally spaced on [0,1]:
#     if z is None:
#         z = np.linspace(0.0, 1.0, len(x))
#
#     # Special case if a single number:
#     # to check for numerical input -- this is a hack
#     if not hasattr(z, "__iter__"):
#         z = np.array([z])
#
#     z = np.asarray(z)
#
#     segments = make_segments(x, y)
#     lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
#                               linewidth=linewidth, alpha=alpha)
#
#     ax = plt.gca()
#     ax.add_collection(lc)
#
#     return lc
#
# def make_segments(x, y):
#     """
#     Create list of line segments from x and y coordinates, in the correct format
#     for LineCollection: an array of the form numlines x (points per line) x 2 (x
#     and y) array
#     """
#
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     return segments
# # multicolored_lines(idx=idxx[0],x=wave_exp, y=flx_exp)
