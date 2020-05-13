#!/usr/bin/env python3
from data_proc_lib import *
t_i= time.time()
# Loading table with info of the data
file= 'spObj-0266-51630-23.fit'
data= Table.read(file)

# Selecting only galaxies
gal_mask= data['objtype']=='GALAXY'
galaxies= data[gal_mask]
galaxies.write('gals-spObj-0266-51630-23.fit', format='fits')


# Removing non-galaxies from directory
dir= 'spSpec/'
head = 'spSpec-51630-0266-'
for object in data:
    if object['objtype'] != 'GALAXY':
        id= object['fiberid']
        file= f'{dir}{head}{id:03}.fit'
        if os.path.exists(file):
            os.remove(file)
        else: print(file, 'file does not exist!')

# List of files

files= []

for galaxy in galaxies:
    id= galaxy['fiberid']
    file= f'{head}{id:03}.fit'
    files.append(file)

n_pixels= 3834
data= np.zeros((len(files),2,n_pixels))
idx= 0
for file in files:
    # Obtaining the redshift
    hdul = fits.open(dir+file)
    Z = hdul[0].header['Z']

    # wavelengths converted to rest frame

    COEFF0 = hdul[0].header['COEFF0']
    COEFF1 = hdul[0].header['COEFF1']

    wavelengths= []

    for i in range(n_pixels):
        lbd= 10**(COEFF0+COEFF1*i)
        wavelengths.append(lbd)
    wavelengths=np.array(wavelengths)/(1+Z)
    w_i= wavelengths[0]
    w_f= wavelengths[-1]
    print(f'{w_i:.2f}',f'{w_f:.2f}')
    # Mean value of the flux
    # First row is the spectrum

    flux = hdul[0].data[0]
    hdul.close()

    median= np.median(flux)
    flux = flux-median
    data[idx]= np.array([wavelengths,flux])
    idx= idx+1

# plt.figure()
# plt.plot(wavelengths,data)
# plt.show()
# plt.close()
t_f= time.time()
print(t_f-t_i)
## The wavelength range is the same, I don't need to computed
# for each file, change that.
