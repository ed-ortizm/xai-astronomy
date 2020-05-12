import os
import numpy as np
from astropy.table import Table

class Data:
    def __init__(self, path='spSpec/'):
        self.path = path
    def files(self):
        # This list store the outputs from walk
        dirpath,dirnames,files = [],[],[]
        # dirpath contains the paths to all the folders that have txt filenames
        # dirnames is a list with only the name of the folders containing the txt filenames
        # files is a list with the names of all the files
        # The map is one to one among each element of these lists, ie, element 1 refer to the same Uumin
        for (a,b,c) in os.walk(self.path):
            dirpath.append(a)
            dirnames.append(b)
            files.append(c)
        return files
