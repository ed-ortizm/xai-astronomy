from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import pickle
from PIL import Image
import shutil
import time

from skimage.segmentation import slic, mark_boundaries
from lime import lime_image
from matplotlib.image import imsave
import numpy as np

from astroExplain.image.imagePlus import GalaxyPlus
from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file = f"image.ini"
parser.read(f"{configuration_file}")
config = ConfigurationFile()
###############################################################################
file_name = parser.get("file", "galaxy")
file_name, file_format = file_name.split(".")
file_location = parser.get("directory", "images")

print(f"Load image: {file_name}.{file_format}", end="\n")

galaxy = Image.open(f"{file_location}/{file_name}.{file_format}")
galaxy = np.array(galaxy, dtype=np.float32)
galaxy *= 1 / galaxy.max()
###############################################################################
print(f"Set explainer configuration", end="\n")
# Load model
base_line = parser.get("model", "base_line")
addGalaxy = GalaxyPlus(base_line=base_line)

slic_configuration = config.section_to_dictionary(parser.items("slic"), [])

segmentation_fn = partial(
    slic,
    n_segments=slic_configuration["segments"],
    compactness=slic_configuration["compactness"],
    sigma=slic_configuration["sigma"],
)

###############################################################################
# Set explainer instance
explainer = lime_image.LimeImageExplainer(random_state=0)
lime_configuration = config.section_to_dictionary(parser.items("lime"), [])

if lime_configuration["hide_color"] == "None":

    explanation_name = f"{file_name}_hide_none"

    lime_configuration["hide_color"] = None

else:

    explanation_name = (
        f"{file_name}_hide_{lime_configuration['hide_color']}"
    )

###############################################################################
# get explanation
print(f"Start explaining {file_name}.{file_format}", end="\n")

explanation = explainer.explain_instance(
    image=galaxy,
    classifier_fn=addGalaxy.predict,
    labels=None,
    hide_color=lime_configuration["hide_color"],
    top_labels=1,
    num_samples=lime_configuration["number_samples"],
    batch_size=lime_configuration["batch_size"],
    segmentation_fn=segmentation_fn,
)

###############################################################################
print(f"Save explanation", end="\n")

save_to = f"{file_location}/{file_name}"
FileDirectory().check_directory(save_to, exit=False)

explanation_name = f"{explanation_name}_base_{base_line}"

with open(f"{save_to}/{explanation_name}.pkl", "wb") as file:

    pickle.dump(explanation, file)
###############################################################################
# save segmented_image
super_pixels = mark_boundaries(
    galaxy,
    explanation.segments,
    color=(1., 1., 1.),
    outline_color=(1., 1., 1.),
)

imsave(f"{save_to}/{file_name}.png", galaxy)
imsave(f"{save_to}/{file_name}_super_pixels.png", super_pixels)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
