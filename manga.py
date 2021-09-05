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
parser.read('manga.ini')
############################################################################
# Relevant data
plate_ifu = parser.get('data', 'plate_ifu')
manga_data_directory = parser.get('directories', 'manga')
(wave,
raw_flux,
stellar_velocity_field) = input_format.get_data(
    plate_ifu=plate_ifu,
    data_directory=manga_data_directory
    )
####################################################################

data_directory = parser.get('directories', 'data')

with fits.open('data/dapall-v2_4_3-2.2.1.fits') as dap_all:

    idx = np.where(dap_all['DAPALL'].data['plateifu'] == plate_ifu)
    z = dap_all['DAPALL'].data['nsa_z'][idx][0]
# ################################################################################
# rest-frame
wave *= 1./(1. + z)
####################################################################
# wavelength master array
number_wave_master = parser.getint('constants', 'wave_master')
wave_master_lower = parser.getint('constants', 'wave_master_lower')
wave_master_upper = parser.getint('constants', 'wave_master_upper')
wave_master = np.linspace(
    wave_master_lower,
    wave_master_upper,
    number_wave_master
    )
####################################################################
flux = input_format.velocity_correction(
    raw_flux=raw_flux,
    wave=wave,
    wave_master=wave_master,
    velocity_field=stellar_velocity_field
    )
np.save('cube.npy', flux)
####################################################################
model = manga.ToyModel(wave_master, delta=5)
explainer = lime_image.LimeImageExplainer()
####################################################################
# image = flux.reshape((1,) + flux.shape)
segmentation_fn = SegmentationAlgorithm('slic', chanel_axis=2)
####################################################################
explanation = explainer.explain_instance(
    flux,
    classifier_fn=model.predict,
    labels=None,#(1,),
    hide_color=0,
    top_labels=1,
    # num_features=100_000,
    num_samples=1000,
    batch_size=10,
    segmentation_fn=segmentation_fn,
    distance_metric='cosine',
    model_regressor=None,
    random_seed=None)
#Select the same class explained on the figures above.
import matplotlib.pyplot as plt
ind =  explanation.top_labels[0]
#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(
    heatmap[0],
    cmap='RdBu',
    vmin=-heatmap.max(),
    vmax=heatmap.max()
    )

plt.colorbar()
plt.show()
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
