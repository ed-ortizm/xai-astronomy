#!/usr/bin/env python3
import os
from time import time

from astropy.table import Table
import numpy as np
import astropy.io.fits as pyfits
from data_proc_lib import spec_cln
from data_proc_lib import get_id
from data_proc_lib import inpl
import matplotlib.pyplot as plt

t_i = time()

# Loading table with info of the data

fname = 'spObj-0266-51630-23.fit'
data = Table.read(fname)

# Number of files with SPEC_CLN = 2

dir = 'spSpec_org/'
head = 'spSpec-51630-0266-'
obj_names = []

for fiber_id in data['fiberid']:
    fname = f'{head}{fiber_id:03}.fit'
    obj_names.append(fname)

gal_count, spec_cln_names = spec_cln(dir=dir, fnames=obj_names)

print(f'There are {gal_count} files with SPEC_CLN = 2')

# Elements with objtype = 'GALAXY'
galaxies = data[np.where(data['objtype'] == 'GALAXY')]
obj_gal_names = []
dir = 'spSpec/'

for fiber_id in galaxies['fiberid']:
    fname = f'{head}{fiber_id:03}.fit'
    obj_gal_names.append(fname)

print(f'There are {len(obj_gal_names)} files with objtype = GALAXY')


# Crossmatching the lists (Venn diagram) with set()

cln_gal = set(spec_cln_names) & set(obj_gal_names)
cln_non_gal = set(spec_cln_names) - set(obj_gal_names)
gal_non_cln = set(obj_gal_names) - set(spec_cln_names)
print(f'{len(cln_gal)} files intersect in the two sets')
print(f'{len(cln_non_gal)} files are SPEC_CLN=2 but not objtype=GALAXY')
print(f'{len(gal_non_cln)} files are objtype=GALAXY but not SPEC_CLN=2')

dir = 'spSpec_org/'
for fname in cln_non_gal:
    fiber_id = get_id(fname)
    #fiber_id = f'{fiber_id:3}'
    #print(data['objtype'][fiber_id]=='GALAXY')

# fiber ids for the intersection
fiber_ids = np.fromiter(map(get_id, cln_gal), dtype=np.int)

# Table of objects in the intersection

# Creating a mask the vectorized way

fiber_id_mask = np.in1d(data['fiberid'], fiber_ids)

# # Creating a mask by brute force:
# m = data['fiberid'].size
# fiber_id_mask = np.zeros(m,dtype=bool)
#
# for fiber_id in fiber_ids:
#     fiber_id_mask[fiber_id-1] = True

gal_itr = data[fiber_id_mask]
gal_itr.write('gal-inter-spObj-0266-51630-23.fit', format='fits',
                overwrite=True)

# Keeping files from the intersection only

dir = 'spSpec_itr/'

for fname in obj_names:
    exist = os.path.exists(dir+fname)
    if exist:
        if fname not in cln_gal:
            os.remove(dir+fname)
    #else:
        #print(f'{fname} file does not exit!')

# Wavelength range: the two coeficients neccessary to compute
# the wavelength range are the same among all the .fit files
# https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it

fname = next(iter(cln_gal))

with pyfits.open(dir+fname) as hdul:
    n_pixels = hdul[0].header['NAXIS1']
    COEFF0 = hdul[0].header['COEFF0']
    COEFF1 = hdul[0].header['COEFF1']

wl_rg = 10. ** (COEFF0 + COEFF1 * np.arange(n_pixels))

# Obtaining redshifts
z = np.array(gal_itr['zfinal'])

# Converting to rest frame

# Using np.newaxis

z_factor = 1/(1+z)
wl_rgs = wl_rg[np.newaxis, :]*z_factor[:, np.newaxis]

# Using outer
# wl_rgs = np.outer(1/(1+z), wl_rg)


# # Or using broadcasting
# nfactor = 1/(1+Z)
# nfactor = nfactor.reshape(nfactor.size,1)
# wl = wl.reshape(1,wl.size)
# wls = nfactor*wl

# Master wavelength range
wl_min = wl_rgs.min()
wl_max = wl_rgs.max()
mtr_wl_rg = np.linspace(wl_min, wl_max, 5_000)

# Fluxes: median removed & interpolated

cln_gal = list(cln_gal)
cln_gal.sort()

flxs_inpl = np.zeros((len(cln_gal), mtr_wl_rg.size))
idx = 0

for fname in cln_gal:
    with pyfits.open(dir+fname) as hdul:
        flux = hdul[0].data[1]
        # median = np.median(flux)
        # flux -= median
        flux = inpl(wl_rgs[idx],flux,mtr_wl_rg)
        flxs_inpl[idx] = flux
    idx += 1

# Saving data to HD

np.save('fluxes_curated',flxs_inpl)
#np.save('master_wavelngth', mtr_wl_rg)

t_f = time()
print(f'The running time is {t_f-t_i:2.5}')

# plt.figure()
# plt.plot(mtr_wl_rg,flxs_inpl[0])
# plt.show()
# plt.close()
## The wavelength range is the same, I don't need to computed
# for each file, change that.
