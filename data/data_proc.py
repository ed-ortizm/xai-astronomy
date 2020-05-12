#!/usr/bin/env python3
from data_proc_lib import *
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
print(len(files))

# Obtaining the redshift

hdul = fits.open(files[0])
Z = hdul[0].header['Z']
hdul.close()
