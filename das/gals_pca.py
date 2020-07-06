#! /usr/bin/env python3

import os
from time import time

import numpy as np
from sklearn.decomposition import PCA
from lib_AE_PCA import plt_spec_pca
from sklearn.preprocessing import StandardScaler

t_i = time()

# Loading the data

dir = 'data/'
fname = 'fluxes_curated.npy'
exist = os.path.exists(dir+fname)

if exist:
    flx = np.load(dir+fname)
else:
    print(f'There is no {fname} in ./{dir}!')

# Performing PCA

pca = PCA(0.99999951) #0.99999951 (100)
tr_flx= pca.fit_transform(flx)
print(f'N° of componets: {pca.n_components_}')

# Inverse transform
inv_tr= pca.inverse_transform(tr_flx)

# Checking the percentages of explained variance
tot_var = sum(pca.explained_variance_)

expl_var = [(i/tot_var)*100 for i in sorted(pca.explained_variance_, reverse=True)]

n = 10
for i in range(n):
    print(f'Component N° {i} explains {expl_var[i]:.3}% of the vatiance')

print(f'These first {n} components explain {sum(expl_var[:n]):.6}% of the variance')

t_f = time()
print(f'The running time is {t_f-t_i:2.5}')

plt_spec_pca(flx[0],inv_tr[0],pca.n_components_)
