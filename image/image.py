"""Explain galaxyPlus model, get super pixel representation and neighbors"""
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import pickle
import time

from matplotlib.image import imsave
import numpy as np
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

from astroExplain.image.imagePlus import GalaxyPlus
from astroExplain.image.explanation import TellMeWhy
from lime import lime_image
from sdss.superclasses import FileDirectory, ConfigurationFile

###############################################################################
START_TIME = time.time()
PARSER = ConfigParser(interpolation=ExtendedInterpolation())
configuration_file = f"image.ini"
PARSER.read(f"{configuration_file}")
config = ConfigurationFile()
###############################################################################
file_name = PARSER.get("file", "galaxy")
file_name, file_format = file_name.split(".")
file_location = PARSER.get("directory", "images")

print(f"Load image: {file_name}.{file_format}", end="\n")

galaxy = Image.open(f"{file_location}/{file_name}.{file_format}")
galaxy = np.array(galaxy, dtype=np.float32)
galaxy *= 1 / galaxy.max()
###############################################################################
print(f"Set explainer configuration", end="\n")
# Load model
base_line = PARSER.get("model", "base_line")
addGalaxy = GalaxyPlus(base_line=base_line)

slic_configuration = config.section_to_dictionary(PARSER.items("slic"), [])

segmentation_fn = partial(
    slic,
    n_segments=slic_configuration["segments"],
    compactness=slic_configuration["compactness"],
    sigma=slic_configuration["sigma"],
)

###############################################################################
# Set explainer instance
explainer = lime_image.LimeImageExplainer(random_state=0)
lime_configuration = config.section_to_dictionary(PARSER.items("lime"), [])

if lime_configuration["hide_color"] == "None":

    explanation_name = f"{file_name}_hide_none"

    lime_configuration["hide_color"] = None

else:

    explanation_name = f"{file_name}_hide_{lime_configuration['hide_color']}"

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

SAVE_TO = f"{file_location}/{file_name}"
FileDirectory().check_directory(SAVE_TO, exit=False)

explanation_name = f"{explanation_name}_base_{base_line}"

with open(
    f"{SAVE_TO}/{explanation_name}_{slic_configuration['segments']}.pkl",
    "wb"
) as file:

    pickle.dump(explanation, file)

# save segmented_image
print(f"Save super pixel representation", end="\n")
super_pixels = mark_boundaries(
    galaxy,
    explanation.segments,
    color=(1.0, 1.0, 1.0),
    outline_color=(1.0, 1.0, 1.0),
)

imsave(f"{SAVE_TO}/{file_name}.png", galaxy)
imsave(f"{SAVE_TO}/{file_name}.pdf", galaxy)
imsave(f"{SAVE_TO}/{file_name}_super_pixels.png", super_pixels)
imsave(f"{SAVE_TO}/{file_name}_super_pixels.pdf", super_pixels)

FINISH_TIME = time.time()
print(f"Run time: {FINISH_TIME-START_TIME:.2f}")
