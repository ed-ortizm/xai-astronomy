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

with open(f"{SAVE_TO}/{explanation_name}.pkl", "wb") as file:

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

# Get neighbors
print(f"Save neighboring galaxies", end="\n")

why = TellMeWhy(explanation)

hide_color = PARSER.get("explain-me", "hide")
hide_color = None if hide_color == "None" else hide_color

neighbors = why.get_neighbors(
    number_samples=PARSER.getint("explain-me", "neighbors"),
    hide_color=hide_color,
)
white_neighbors = why.get_neighbors(
    number_samples=PARSER.getint("explain-me", "neighbors"),
    hide_color=1,
)

neighbors_directory = f"{SAVE_TO}/neighbors"
FileDirectory().check_directory(SAVE_TO, exit=False)

for idx, neighbor in enumerate(neighbors):

    neighbor = mark_boundaries(
        neighbor,
        explanation.segments,
        color=(1.0, 1.0, 1.0),
        outline_color=(0.0, 0.0, 0.0),
    )

    imsave(f"{SAVE_TO}/{idx:03d}_neighbor.pdf", neighbor)
    imsave(f"{SAVE_TO}/{idx:03d}_neighbor.png", neighbor)

    white_neighbor = mark_boundaries(
        white_neighbors[idx],
        explanation.segments,
        color=(1.0, 1.0, 1.0),
        outline_color=(0.0, 0.0, 0.0),
    )

    imsave(
        f"{neighbors_directory}/{idx:03d}_white_neighbor.pdf", white_neighbor
    )
    imsave(
        f"{neighbors_directory}/{idx:03d}_white_neighbor.png", white_neighbor
    )
# show me explanations
show_me_directory = f"{SAVE_TO}/show_me"
FileDirectory().check_directory(show_me_directory, exit=False)
positive_directory = f"{show_me_directory}/positive"
FileDirectory().check_directory(positive_directory, exit=False)
negative_directory = f"{show_me_directory}/negative"
FileDirectory().check_directory(negative_directory, exit=False)
# so select 25% off in a shell super pixels
show_number = int(lime_configuration["number_samples"]/4)
# positive
for idx in range(show_number):

    contribution = why.show_me(
    positive_only=True,
    negative_only=False,
    number_of_features=idx,
    hide_rest=False,
    )

    imsave(f"{positive_directory}/positive_{idx:03d}.pdf", contribution)
    imsave(f"{positive_directory}/positive_{idx:03d}.png", contribution)

# negative
for idx in range(show_number):

    contribution = why.show_me(
    positive_only=False,
    negative_only=True,
    number_of_features=idx,
    hide_rest=False,
    )

    imsave(f"{negative_directory}/negative_{idx:03d}.pdf", contribution)
    imsave(f"{negative_directory}/negative_{idx:03d}.png", contribution)
# green and Red

# finish script
FINISH_TIME = time.time()
print(f"Run time: {FINISH_TIME-START_TIME:.2f}")
