#!/usr/bin/env python3
import numpy as np
from astropy.table import Table

# Loading data
file= 'spObj-0266-51630-23.fit'
data= Table.read(file)

# Selecting only galaxies
gal_mask= data['objtype']=='GALAXY'
galaxies = data[gal_mask]
