#!/usr/bin/env python3

import os
import sys
from time import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from AEs_lib import plot_2D, plt_spec_pca
from AEs_lib import AEpca, PcA


ti = time()

# Loading the data
print('Loading the data...')
proc_dbPath = f'/home/edgar/zorro/SDSSdata/data_proc'
fname =  f'flxs_5000_div_med.npy'
dest = f'{proc_dbPath}/{fname}'
if os.path.exists(dest):
    spec = np.load(dest, mmap_mode='r')
else:
    print(f'There is no {fname} in {proc_dbPath} directory!')

### Performing PCA
print('Performing PCA...')

pca = PcA()

tr_spec = pca.fit(spec)

## Inverse transform
#print('Performing inverse PCA...')

#inv_tr = pca.inverse(tr_flxs)
#plot_2D(tr_flxs, 'PCA')

## Ploting a spetrum
#plt_spec_pca(flxs[0],inv_tr[0],pca.n_components)

## Normalizing the flux: removing the mean value and normalizing
# by the standard deviation. This is done because
# pca.fit_transform does the same on the data. Therefore if we
# want to compare we need to have the same data.

#sc = StandardScaler()
#flxs = sc.fit_transform(flxs)

## Creating the AE
print('Training Auto Encoder...')

AE = AEpca(in_dim=spec.shape[1], epochs=10)
AE.fit(spec)
#pred = AE.predict(flxs)
#plt_spec_pca(flxs[-1,:],pred[-1,:],2)
codings = AE.encode(spec)
#plot_2D(codings, 'AE')

tf = time()
print(f'Running time: {tf-ti:.2f}')
