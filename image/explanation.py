"""Explain galaxyPlus model, get super pixel representation and neighbors"""
from configparser import ConfigParser, ExtendedInterpolation
import pickle
import time

import matplotlib.pyplot as plt
from matplotlib.image import imsave
import numpy as np
from skimage.segmentation import mark_boundaries

from astroExplain.image.explanation import TellMeWhy
from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
START_TIME = time.time()
PARSER = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file = f"explanation.ini"
PARSER.read(f"{configuration_file}")
config = ConfigurationFile()
###############################################################################
file_name = PARSER.get("file", "explanation")
file_name, file_format = file_name.split(".")
file_location = PARSER.get("directory", "explanation")

print(f"Load explanation: {file_name}.{file_format}", end="\n")

with open(f"{file_location}/{file_name}.{file_format}", "rb") as file:

    explanation = pickle.load(file)
    why = TellMeWhy(explanation)

# save segmented_image
print(f"Save super pixel representation", end="\n")

super_pixels = mark_boundaries(
    explanation.image,
    explanation.segments,
    color=(1.0, 1.0, 1.0),
    outline_color=(1.0, 1.0, 1.0),
)

imsave(f"{file_location}/{file_name}.png", explanation.image)
imsave(f"{file_location}/{file_name}.pdf", explanation.image)
imsave(f"{file_location}/{file_name}_super_pixels.png", super_pixels)
imsave(f"{file_location}/{file_name}_super_pixels.pdf", super_pixels)

# Get neighbors
print(f"Save neighboring galaxies", end="\n")

hide_color = PARSER.get("explain-me", "hide")
hide_color = None if hide_color == "None" else hide_color

neighbors = why.get_neighbors(
    number_samples=PARSER.getint("explain-me", "neighbors"),
    hide_color=hide_color,
)

neighbors_directory = f"{file_location}/neighbors"
FileDirectory().check_directory(neighbors_directory, exit=False)

for idx, neighbor in enumerate(neighbors):

    neighbor = mark_boundaries(
        neighbor,
        explanation.segments,
        color=(1.0, 1.0, 1.0),
        outline_color=(0.0, 0.0, 0.0),
    )

    imsave(f"{neighbors_directory}/{idx:03d}_neighbor.pdf", neighbor)
    imsave(f"{neighbors_directory}/{idx:03d}_neighbor.png", neighbor)

# show me explanations
show_me_directory = f"{file_location}/show_me"
FileDirectory().check_directory(show_me_directory, exit=False)

positive_directory = f"{show_me_directory}/positive"
FileDirectory().check_directory(positive_directory, exit=False)
# so select 25% off in a shell super pixels
show_number = PARSER.getint("explain-me", "show")
# positive
for idx in range(show_number):

    number_of_features = 2**(idx+1)

    contribution, contribution_mask = why.show_me(
    positive_only=True,
    negative_only=False,
    number_of_features=number_of_features,
    hide_rest=False,
    )

    contribution = mark_boundaries(
        contribution,
        contribution_mask,
        color=(0, 0, 0),
        outline_color=(1.0, 1.0, 1.0),
    )

    imsave(f"{positive_directory}/positive_{2**(idx+1):03d}.pdf", contribution)
    imsave(f"{positive_directory}/positive_{2**(idx+1):03d}.png", contribution)

# green and Red
both_directory = f"{show_me_directory}/both"
FileDirectory().check_directory(both_directory, exit=False)

for idx in range(show_number+1):

    number_of_features = 2**(idx+1)

    contribution, contribution_mask = why.show_me(
    positive_only=False,
    negative_only=False,
    number_of_features=number_of_features,
    hide_rest=False,
    )

    contribution = mark_boundaries(
        contribution,
        contribution_mask,
        color=(0, 0, 0),
        outline_color=(1.0, 1.0, 1.0),
    )

    imsave(f"{both_directory}/both_{2**(idx+1):03d}.pdf", contribution)
    imsave(f"{both_directory}/both_{2**(idx+1):03d}.png", contribution)

# Get heatmap
heatmap = why.get_heatmap()
vmax = np.nanmax(heatmap)
#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -vmax, vmax = vmax)
plt.colorbar()
plt.savefig(f"{file_location}/heatmap.png")
plt.savefig(f"{file_location}/heatmap.pdf")

# finish script
FINISH_TIME = time.time()
print(f"Run time: {FINISH_TIME-START_TIME:.2f}")
