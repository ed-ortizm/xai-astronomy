from configparser import ConfigParser, ExtendedInterpolation
import time
import pickle
import os

import numpy as np
from skimage.segmentation import felzenszwalb, quickshift, slic
from PIL import Image

from sdss.superclasses import ConfigurationFile, FileDirectory

###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("segment.ini")
config = ConfigurationFile()
###############################################################################
# Load image as np.array
input_directory = parser.get("directory", "images")
file_name = parser.get("file", "galaxy")
name_galaxy, in_format = file_name.split(".")

print("Load {name_galaxy} image", end="\n")

if in_format == "npy":

    galaxy = np.load(f"{input_directory}/{file_name}")

else:
    galaxy = np.array(
        Image.open(f"{input_directory}/{file_name}"), dtype=float
    )

# normalize pixel space
galaxy *= 1 / galaxy.max()

###############################################################################
algorithms = parser.get("common", "algorithms")
algorithms = config.entry_to_list(algorithms, entry_type=str, separator="\n")

#     if segmentation == "quickshift":
# def segment_image(img, ratio=1., max_dist=200, kernel_size=5):
#      ...:
#      ...:     segments = quickshift(img, ratio=ratio, max_dist=max_dist, kernel_size=kernel_size)
#      ...:
#      ...:     segmented_image = mark_boundaries(img, segments)
#      ...:
#      ...:     clf()
#      ...:     imshow(segmented_image)
#      ...:
#      ...:     n_segments = np.unique(segments).size
#      ...:     print(f"{n_segments} segments")
#      ...:
#      ...:     return segmented_image, segments
#         segmented_image, segments = segment_image(img, ratio=0.3, max_dist=200, kernel_size=10)
#     a = np.random.choice(np.unique(segments), size=np.random.randint(low=0, high=segments.max()))
#      mask = np.zeros(segments.shape, dtype=bool)
#      ...: for val in a:
#      ...:     mask |= segments == val
#
#      masked = img.copy()
#      ...: masked[~mask] = 0
#      ...: clf()
#      ...: imshow(masked)
#
#
# In [212]: plt.title("Neighboring Meaningful Data Representation")
# Out[212]: Text(0.5, 1.0, 'Neighboring Meaningful Data Representation')
#
# In [213]: plt.xticks([]), plt.yticks([])
# Out[213]: (([], []), ([], []))
#
# In [214]: plt.tight_layout()
#
# In [215]: plt.savefig("xAI/eyeSegementedPerturbation04.pdf")
#
#     elif segmentation == "slic":
#
#         pass
#
#     elif segmentation == "felzenszwalb":
#
#         pass
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
