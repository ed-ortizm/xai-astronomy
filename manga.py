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
from lime import lime_tabular
####################################################################
import numpy as np
import scipy.constants as cst
################################################################################
from library_outlier import Outlier
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

flux = np.empty(
            (
            number_x*number_y,
            number_wave_master
            )
        )

raw_flux = raw_flux.reshape(
    number_x*number_y,
    number_z
    )

for idx, stellar_velocity in enumerate(stellar_velocity_field.reshape(-1)):

    flux[idx, :] = np.interp(
        wave_master,
        wave*cst.c/(cst.c + stellar_velocity*1_000),
        raw_flux[idx, :],
        left=np.nan,
        right=np.nan
    )

# flux *= 1. / np.nanmedian(flux, axis=0)
flux = flux.reshape(number_x, number_y, number_wave_master)
print(flux.shape, flux.min(), flux.max())



# all_flx[:, idx] = np.interp(m_wl, wl*cst.c / (cst.c + stellar_vel[idx]*1_000.),
    # flx[:, idx], left=np.nan, right=np.nan)

    # all_flx *= 1. / np.nanmedian(all_flx, axis=0)
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
