#!/usr/bin/env python3
import os
from astropy.table import Table
import numpy as np
from astropy.io import fits
from data_proc_lib import spec_cln
from data_proc_lib import get_id
#t_i= time.time()
# Do a catalog with the ones that are both 'GALAXY' and SPEC_CLN=2
# Set apart the ones that are 'GALAXY' but not SPEC_CLN=2
# Set apart the ones  that are SPEC_CLN=2 but not 'GALAXY'

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

gal_count,spec_cln_names = spec_cln(dir=dir,fnames=obj_names)
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

# fiber ids for the intersection
fiber_ids = np.fromiter(map(get_id,cln_gal),dtype=np.int)

# Table of objects in the intersection

# Creating a mask by brute force:
m = data['fiberid'].size
fiber_id_mask = np.zeros(m,dtype=bool)

for fiber_id in fiber_ids:
    fiber_id_mask[fiber_id-1] = True

gal_itr = data[fiber_id_mask]
gal_itr.write('gal-inter-spObj-0266-51630-23.fit', format='fits',
                overwrite=True)

# Saving intersection set

dir = 'spSpec_itr/'

for fname in obj_names:
    exist = os.path.exists(dir+fname)
    if exist:
        if fname not in cln_gal:
            os.remove(dir+fname)
    else:
        print(f'{fname} file does not exit!')

#
# for fname in cln_gal:
#     exist = os.path.exists(dir+fname)
#     if exist:

# Wavelength range: the two coeficients neccessary to compute
# the wavelength range are the same among all the .fit files
# https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it

fname = next(iter(cln_gal))
with fits.open(dir+fname) as hdul:
    n_pixels = hdul[0].header['NAXIS1']
    COEFF0 = hdul[0].header['COEFF0']
    COEFF1 = hdul[0].header['COEFF1']

wl_rg = 10. ** (COEFF0 + COEFF1 * np.arange(n_pixels))

# Obtaining redshifts
Z = np.array(gal_itr['zfinal'])

# Converting to rest frame
wl_rgs = np.outer(1/(1+Z),wl_rg)

# # Or using broadcasting
# nfactor = 1/(1+Z)
# nfactor = nfactor.reshape(nfactor.size,1)
# wl = wl.reshape(1,wl.size)
# wls = nfactor*wl

# Master wavelength range
wl_min = wl_rgs.min()
wl_max = wl_rgs.max()
mtr_wl_rg = np.linspace(wl_min,wl_max,5_000)

# Fluxes: median removed
fluxes = np.zeros(wl_rgs.shape)
idx = 0
for fname in cln_gal:
    with fits.open(dir+fname) as hdul:
        flux = hdul[0].data[0]
        median = np.median(flux)
        flux -= median
        fluxes[idx] = flux
    idx += 1

# # Interpolating fluxes
#
# m= raw_data.shape[0]
# n= m_wavelength_range.shape[0] # Do a sampling where you have a similar
# # resolution as in al the wavelength ranges. (consider higest redshift)
# cured_data= np.zeros((m,n))
# #
# for i in range(m):
#     wavelength= raw_data[i,0,:]
#     flux= raw_data[i,1,:]
#     cured_data[i]= f_interpolate(wavelength,flux,m_wavelength_range)
#     print(cured_data.size*cured_data.itemsize)
    # Killed
    ## Trying paralell
    # with mp.Pool() as pool:
    #     f=\
    #     pool.starmap(\
    #     p_f_interpolate,zip(wavelength,flux)\
    #     )
    #     # in line 97: TypeError: object of type
    #     #'numpy.float64' has no len()
    # cured_data[i]= f(m_wavelength_range)
#
# plt.figure()
# plt.plot(wavelengths,data)
# plt.show()
# plt.close()
#t_f= time.time()
#print(t_f-t_i)
## The wavelength range is the same, I don't need to computed
# for each file, change that.
