#!/usr/bin/env python3

import os
import sys
from time import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from AEs_lib import plot_2D, plt_spec_pca
from AEs_lib import AEpca, PcA, Outlier


ti = time()

# Loading the data
print('Loading the data...')
path = f'/home/edgar/zorro/outlier_AEs'
fname =  f'spec_99356.npy'
dest = f'{path}/{fname}'
if os.path.exists(dest):
    spec = np.load(dest, mmap_mode='r')
else:
    print(f'There is no {fname} in {path} directory!')

### Performing PCA
#print('Performing PCA...')
#
#pca = PcA(n_comps=2)
#pca.fit(spec)
#latent_pca = pca.predict(spec)
#np.save('latent_pca.npy', latent_pca)
#
## Creating the AE
#print('Training Auto Encoder...')
#
#AE = AEpca(in_dim=spec.shape[1], epochs=30)
#AE.fit(spec)
#
## Saving model
#AE.save()
#
## Latent space
#latent_AE = AE.encode(spec)
#np.save('latent_AE.npy', latent_AE)
#
## Saving AE predictions
#pred = AE.predict(spec)
#np.save('pred_AE.npy', pred)


pred = np.load('pred_AE.npy')
# Outlier scores
outlier = Outlier(N=10)

# Area
outlier.area(spec, pred)

# chi2
outlier.chi2(spec, pred)

# mse
outlier.mse(spec, pred)

# mad
outlier.mad(spec, pred)

# lp 2, 1, 0.5, 0.3
outlier.lp(spec, pred, p=2)
outlier.lp(spec, pred, p=1)
outlier.lp(spec, pred, p=0.5)
outlier.lp(spec, pred, p=0.3)

# lpf 2, 1, 0.5, 0.3
outlier.lpf(spec, pred, p=2)
outlier.lpf(spec, pred, p=1)
outlier.lpf(spec, pred, p=0.5)
outlier.lpf(spec, pred, p=0.3)

tf = time()
print(f'Running time: {tf-ti:.2f}')
