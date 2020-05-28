#! /usr/bin/env python3

import numpy as np
import os
from sklearn.decomposition import PCA
from time import time
from lib_AE_PCA import plt_spec_pca
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
pca = PCA(0.99999951)

tr_flx= pca.fit_transform(flx)
print(f'N° of componets: {pca.n_components_}')

# Inverse transform
inv_tr= pca.inverse_transform(tr_flx)

t_f = time()
print(f'The running time is {t_f-t_i:2.5}')

plt_spec_pca(flx[0],inv_tr[0],pca.n_components_)
