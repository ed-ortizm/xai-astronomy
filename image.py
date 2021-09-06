#! /usr/bin/env python3
################################################################################
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import sys
import time
################################################################################
from astropy.io import fits
####################################################################
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
####################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
################################################################################
from src.explainers.manga import manga
from src.explainers.manga import input_format
################################################################################
ti = time.time()
################################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('image.ini')
############################################################################
# Relevant data
plate_ifu = parser.get('data', 'plate_ifu')
manga_data_directory = parser.get('directories', 'manga')

flux = np.load(f'{manga_data_directory}/{plate_ifu}_image.npy')
####################################################################
# wave_master = np.linspace(3500, 7500, 4001)
model = manga.ToyModel(wave=None, cube=False)
explainer = lime_image.LimeImageExplainer()
####################################################################
# segmentation_fn = SegmentationAlgorithm('slic', chanel_axis=2)
####################################################################
explanation = explainer.explain_instance(
    flux,
    classifier_fn=model.predict,
    labels=(1,),
    hide_color=0,
    top_labels=1,
    # num_features=100_000,
    num_samples=10_000,
    batch_size=100,
    # segmentation_fn=segmentation_fn,
    # distance_metric='cosine',
    # model_regressor=None,
    random_seed=None
    )
# #Select the same class explained on the figures above.
ind =  explanation.top_labels[0]
#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
# #Plot. The visualization makes more sense if a symmetrical colorbar is used.
# plt.imshow(
#     heatmap[:],
#     cmap='RdBu',
#     vmin=-heatmap.max(),
#     vmax=heatmap.max()
#     )
#
# plt.colorbar()
# plt.show()
################################################################################
from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=6,
    hide_rest=False)
# plt.imshow(mark_boundaries(temp, mask))
# plt.show()
################################################################################
fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)

ax[0].imshow(flux)
ax[0].set_title('GALAXY')

ax[1].imshow(mark_boundaries(temp, mask))
ax[1].set_title('Explanation')

ax[2].imshow(
    heatmap[:],
    cmap='RdBu',
    vmin=-heatmap.max(),
    vmax=heatmap.max()
    )
ax[2].set_title('Heatmap')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
# plt.colorbar()

plt.show()
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
