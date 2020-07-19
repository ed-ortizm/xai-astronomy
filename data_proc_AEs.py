#! /usr/bin/env python3

import os
from time import time

import numpy as np
import pandas as pd

from AEs_lib import spectra

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

n_obs = 10
dbPath = f'{os.getcwd()}/db'
fname = f'gs_{n_obs}.csv'
dest = f'{dbPath}/{fname}'
gs = pd.read_csv(dest)

m_wl, flxs = spectra(gs, dbPath)

np.save(f'{dbPath}/flxs_{flxs.shape[0]}.npy', flxs)
np.save(f'{dbPath}/wl_grid_{m_wl.size}.npy', m_wl)
#
# # for flx in flxs:
# #     plt.figure()
# #     plt.plot(m_wl, flx)
#     plt.show()
#     plt.close()
