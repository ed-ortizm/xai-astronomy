#! /usr/bin/env python3

import os
from time import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from proc_sdss_lib import spectra

ti = time()
# Data processing

# For some reason when I check for SPEC_CLN with fitsheader, it is not there.
#
# Computing the wavelength range
# plate = gs['plate'][0]
# mjd = gs['mjd'][0]
# fiberid = gs['fiberid'][0]
# run2d = gs['run2d'][0]
# z = gs['z'][0]
#
# min, max, flx = min_max_interp(plate, mjd, fiberid, run2d, z, dbPath)
#
# print(f'min= {min:.2f}, max= {max:.2f}')

## Loading DataFrame with the data of the galaxies

dbPath = f'/home/edgar/zorro/SDSSdata'
gs = pd.read_csv(f'{dbPath}/gs_SN_median_sorted.csv')


n_obs = 50_000 # 3188712

if not os.path.exists(f'{dbPath}/gs_{n_obs}.csv'):
    print(f'Creating file: gs_{n_obs}.csv')
    fname = f'gs_{n_obs}.csv'
    dest = f'{dbPath}/{fname}'
    gs_n = gs[:n_obs]
    gs_n.index = np.arange(n_obs)
    gs_n.to_csv(f'{dbPath}/gs_{n_obs}.csv')
    print('Starting data curation process...')
    gs_n = pd.read_csv(f'{dbPath}/gs_{n_obs}.csv')
    m_wl, flxs = spectra(gs_n, dbPath)
    np.save(f'{dbPath}/data_proc/flxs_{flxs.shape[0]}.npy', flxs)
    np.save(f'{dbPath}/data_proc/wl_grid_{m_wl.size}.npy', m_wl)
else:
    print('Starting data curation process...')
    gs_n = pd.read_csv(f'{dbPath}/gs_{n_obs}.csv')
    m_wl, flxs = spectra(gs_n, dbPath)
    np.save(f'{dbPath}/data_proc/flxs_{flxs.shape[0]}.npy', flxs)
    np.save(f'{dbPath}/data_proc/wl_grid_{m_wl.size}.npy', m_wl)

#for flx in flxs:
#   plt.figure()
#   plt.plot(m_wl, flx)
#   plt.show()
#   plt.close()
tf = time()

print(f'Running time: {tf-ti:.2f} [seg]')
