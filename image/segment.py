"""Get segments of an image using SLIC from skimage"""
from configparser import ConfigParser, ExtendedInterpolation
import shutil
import time

from PIL import Image
from matplotlib.image import imsave
import numpy as np
from skimage.segmentation import slic, mark_boundaries

from sdss.utils.configfile import ConfigurationFile
from sdss.utils.managefiles import FileDirectory

###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file_name = "segment.ini"
parser.read(f"{configuration_file_name}")
config = ConfigurationFile()
###############################################################################
file_name = parser.get("file", "galaxy")
file_location = parser.get("directory", "images")

print(f"Load image: {file_name}", end="\n")

galaxy = Image.open(f"{file_location}/{file_name}")
galaxy = np.array(galaxy, dtype=float)
galaxy *= 1 / galaxy.max()
###############################################################################
print(f"Load configuration of SLIC segmetation algorithms", end="\n")

slic_configuration = config.section_to_dictionary(parser.items("slic"), [])

galaxy_segments = slic(
    galaxy,
    n_segments=slic_configuration["segments"],
    compactness=slic_configuration["compactness"],
    sigma=slic_configuration["sigma"],
)
super_pixels = mark_boundaries(galaxy, galaxy_segments)

###############################################################################
name_super_pixels = file_name.split(".")[0]

save_to = f"{file_location}/{name_super_pixels}"
FileDirectory().check_directory(save_to, exit_program=False)

output_file_name = (
    f"{name_super_pixels}_"
    f"compactness_{slic_configuration['compactness']}_"
    f"{slic_configuration['segments']}_super_pixels.png"
)
imsave(f"{save_to}/{name_super_pixels}.png", galaxy)
imsave(f"{save_to}/{output_file_name}", super_pixels)
###############################################################################
print(f"Save configuration file", end="\n")
shutil.copyfile(
    f"./{configuration_file_name}", f"{save_to}/{configuration_file_name}"
)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
