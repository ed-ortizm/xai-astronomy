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
fnames = []

for fiber_id in data['fiberid']:
    fname = f'{head}{fiber_id:03}.fit'
    fnames.append(fname)

gal_count,spec_cln_names = spec_cln(dir=dir,fnames=fnames)
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

# fiber_ids having the same number of elements of the original table

# non_galaxies = data[np.where(data['objtype'] != 'GALAXY')]
# non_galaxies.write('non-gals-spObj-0266-51630-23.fit', format='fits',
#                    overwrite=True)
#

#
# # Wavelength range: the two coeficients neccessary to compute
# # the wavelength range are the same among all the .fit files
#
# with fits.open(dir+fnames[0]) as hdul:
#     n_pixels = hdul[0].header['NAXIS1']
#     COEFF0 = hdul[0].header['COEFF0']
#     COEFF1 = hdul[0].header['COEFF1']
#
# wl_range = 10. ** (COEFF0 + COEFF1 * np.arange(n_pixels))

# double check if the vacumn ws are the same.... if so, do (w)/(1+Z)
# where Z is an array (broadcasting) double check if you can take then
# from the table.
# for i in range(n_pixels):
#     lbd= 10**(COEFF0+COEFF1*i)
#     wavelength_range[i]= lbd
# hdul.close()
# # wavelength_range = 10. ** (COEFF0 + COEFF1 * np.arange(n_pixels))
#
# # Data with wavelengths converted to rest frame
# # Master wavelength range
#
# raw_data= np.zeros((len(files),2,n_pixels))
#
# idx= 0
# for file in files:
#     # Obtaining the redshift
#     hdul = fits.open(dir+file)
#     Z = hdul[0].header['Z']
#
#     wavelengths_rest_frame= np.array(wavelength_range)/(1+Z)
#     if idx==0:
#         m_wavelength_range= wavelengths_rest_frame
#     else:
#         m_wavelength_range=\
#         np.concatenate((m_wavelength_range,wavelengths_rest_frame))
#         # create a list, it is more efficient with memory
#
#     # w_i= wavelengths_rest_frame[0]
#     # w_f= wavelengths_rest_frame[-1]
#     # print(f'{w_i:.2f}',f'{w_f:.2f}')
#
#     # Mean value of the flux
#     # First row is the spectrum
#
#     flux = hdul[0].data[0]
#     hdul.close()
#
#     median= np.median(flux)
#     flux = flux-median# Stay in place!!
#     raw_data[idx]= np.array([wavelengths_rest_frame,flux])
#     idx= idx+1
#
# # Master wavelength range
# m_wavelength_range= np.unique(m_wavelength_range)
# print(m_wavelength_range.shape,len(m_wavelength_range))
# # w1= m_wavelength_range[0]
# # w2= m_wavelength_range[-1]
# # print(w1,w2)
#
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
