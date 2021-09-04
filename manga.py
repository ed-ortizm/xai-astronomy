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
from src.explainers.manga.manga import ToyModel
################################################################################
ti = time.time()
################################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('manga.ini')
############################################################################
# Relevant data
plate_ifu = parser.get('data', 'plate_ifu')
with fits.open(
    'data/manga/manga-7443-12703-LOGCUBE-HYB10-GAU-MILESHC.fits.gz'
    ) as cube:

    # Re-order FLUX, IVAR, and MASK arrays
    # (wavelength, DEC, RA) to (RA, DEC, wavelength)

    wave = cube['wave'].data
    raw_flux = np.transpose(cube['FLUX'].data, axes=(2, 1, 0))
    ivar = np.transpose(cube['IVAR'].data, axes=(2, 1, 0))
    mask = np.transpose(cube['MASK'].data, axes=(2, 1, 0))
    # get units
    flux_header = cube['FLUX'].header

with fits.open(
    'data/manga/manga-7443-12703-MAPS-HYB10-GAU-MILESHC.fits.gz'
    ) as map:

    stellar_velocity_field = map['stellar_vel'].data

with fits.open('data/dapall-v2_4_3-2.2.1.fits') as dap_all:

    ind = np.where(dap_all['DAPALL'].data['plateifu'] == plate_ifu)
    z = dap_all['DAPALL'].data['nsa_z'][ind][0]
################################################################################
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
# Correcting for the stellar velocity redshift
# print(stellar_velocity.shape, flux.shape)
number_x, number_y, number_z = raw_flux.shape

flux = np.empty( (number_x*number_y, number_wave_master) )

raw_flux = raw_flux.reshape(number_x*number_y, number_z)

for idx, stellar_velocity in enumerate(stellar_velocity_field.reshape(-1)):

    flux[idx, :] = np.interp(
        wave_master,
        wave*cst.c/(cst.c + stellar_velocity*1_000),
        raw_flux[idx, :],
        left=np.nan,
        right=np.nan
    )

    median = np.nanmedian(flux[idx, :])
    if median != 0:
        flux[idx, :] *= 1./median
####################################################################
flux = flux.reshape(number_x, number_y, number_wave_master)
np.save('cube.npy', flux)
####################################################################
model = ToyModel(wave_master, delta=10)
explainer = lime_image.LimeImageExplainer()
####################################################################
image = flux.reshape((1,) + flux.shape)
print(image.shape)
# image = np.repeat(flux[:, :, :, np.newaxis], 3, axis=3)
# image[..., 1:] = 0.
segmentation_fn = SegmentationAlgorithm('slic')
# , kernel_size=4,
#                                                     max_dist=200, ratio=0.2,
#                                                     random_seed=random_seed)
explanation = explainer.explain_instance(
    image,
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
plt.imshow(heatmap[0], cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
