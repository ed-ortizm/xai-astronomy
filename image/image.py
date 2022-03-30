from configparser import ConfigParser, ExtendedInterpolation
import pickle
import time

from matplotlib.image import imsave
import numpy as np
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

from sdss.superclasses import ConfigurationFile, FileDirectory
###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file_name = "image.ini"
parser.read(f"{configuration_file_name}")
config = ConfigurationFile()
###############################################################################
file_name = parser.get("file", "galaxy")
file_location = parser.get("directory", "images")

print(f"Load image: {file_name}", end="\n")

galaxy = Image.open(f"{file_location}/{file_name}")
galaxy = np.array(galaxy, dtype=float)
galaxy *= 1/galaxy.max()
###############################################################################
print(f"Load configuration of SLIC segmetation algorithms", end="\n")

slic_configuration = config.section_to_dictionary(parser.items("slic"), [])

galaxy_segments = slic(
    galaxy,
    n_segments=slic_configuration["segments"],
    compactness=slic_configuration["compactness"],
    sigma=slic_configuration["sigma"]
)
super_pixels = mark_boundaries(galaxy, galaxy_segments)

name_super_pixels = file_name.split(".")[0]
name_super_pixels = f"{name_super_pixels}_SLIC.png"

imsave(f"{file_location}/{name_super_pixels}", super_pixels)
###############################################################################
# print(f"Save configuration file", end="\n")
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
