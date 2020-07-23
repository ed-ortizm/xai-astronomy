#!/usr/bin/env python3

import os
import sys
from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from AEs_lib import plot_2D
from AEs_lib import plt_spec_pca


ti = time()

## Loading the data

proc_dbPath = f'/home/edgar/zorro/SDSSdata/data_proc'
fname =  f'flxs_1000.npy'
dest = f'{proc_dbPath}/{fname}'
if os.path.exists(dest):
    flxs = np.load(dest, mmap_mode='r')
else:
    print(f'There is no {fname} in {proc_dbPath} directory!')

## Performing PCA
print('Performing PCA...')

#pca = PCA(n_components=2) #0.99999951 (100)
#
#nnans = np.count_nonzero(np.isnan(flxs))
#print(f'Number of nans is {nnans}')
#tr_flxs= pca.fit_transform(flxs)

#print(f'N° of componets: {pca.n_components_}, {tr_flxs.shape}')

## Inverse transform

#inv_tr= pca.inverse_transform(tr_flxs)

# ## Checking the percentages of explained variance
#
# tot_var = sum(pca.explained_variance_)
#
# expl_var = [(i/tot_var)*100 for i in sorted(pca.explained_variance_, reverse=True)]

#plot_2D(tr_flxs, 'PCA')

# n = 10
# for i in range(n):
#     print(f'Component N° {i} explains {expl_var[i]:.3}% of the vatiance')
#
# print(f'These first {n} components explain {sum(expl_var[:n]):.6}% of the variance')

## Ploting a spetrum

#plt_spec_pca(flxs[0],inv_tr[0],pca.n_components_)


## Creating the AE
# I do create it in two parts because later on I'll need to acces the latent
# space. Therefore it is necessary to add the input and output shape

print('Training Auto Encoder...')
encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=(flxs.shape[1],))])
decoder = keras.models.Sequential([keras.layers.Dense(flxs.shape[1])])
autoencoder = keras.models.Sequential([encoder, decoder])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
#optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
autoencoder.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

## Normalizing the flux: removing the mean value and normalizing by the standard
# deviation. This is done because pca.fit_transform does the same on the data.
# Therefore if we want to compare we need to have the same data.

sc = StandardScaler()
flxs = sc.fit_transform(flxs)

## Fitting

history = autoencoder.fit(flxs, flxs, epochs=100)
latent = encoder.predict(flxs)

print(latent.shape)

plot_2D(latent, 'AE')

tf = time()
print(f'Running time: {tf-ti:.2f}')
