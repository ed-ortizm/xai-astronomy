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
################################################################################
from library_outlier import Outlier
################################################################################
ti = time.time()
################################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('manga.ini')
############################################################################
# loading cube from marvin
plate_ifu = parser.get('data', 'plate_ifu')
cube = Cube(plateifu=plate_ifu)
with fits.open(
    'data/manga/manga-7443-12703-LOGCUBE-HYB10-GAU-MILESHC.fits.gz'
    ) as cube:

    wave = cube['wave'].data
    flux = cube['flux'].data

with fits.open(
    'data/manga/manga-7443-12703-MAPS-HYB10-GAU-MILESHC.fits.gz'
    ) as map:

    stellar_velocity = map['stellar_vel'].data

with fits.open('data/dapall-v2_4_3-2.2.1.fits') as dap_all:

    ind = np.where(dapall['DAPALL'].data['plateifu'] == plate_ifu)
    z = dapall['DAPALL'].data['nsa_z'][ind][0]

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
