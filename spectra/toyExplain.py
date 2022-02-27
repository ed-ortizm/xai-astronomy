from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import time
import pickle
import os

from lime import lime_image
import numpy as np

from astroExplain.segmentation import SpectraSegmentation
from astroExplain.toyRegressors import SpectraPlus
from sdss.superclasses import FileDirectory

###############################################################################
start_time = time.time()
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("toyExplain.ini")
###############################################################################
# Load image as np.array
print("Load spectrum", end="\n")

input_directory = parser.get("directory", "input")
file_name = parser.get("file", "galaxy")
name_galaxy, in_format = file_name.split(".")

galaxy = np.load(f"{input_directory}/{file_name}")
###############################################################################
output_directory = parser.get("directory", "output")

# Use instead FileDirectory
if os.path.exists(output_directory) is False:
    os.makedirs(output_directory)

# Load model
addSpectra = SpectraPlus()

# Set explainer instance
explainer = lime_image.LimeImageExplainer(random_state=0)

number_segments = parser.getint("lime", "number_segments")
segmentation_fn = SpectraSegmentation().uniform
segmentation_fn = partial(segmentation_fn, number_segments=number_segments)
# get explanation

explanation = explainer.explain_instance(
    image=galaxy[np.newaxis, ...], # image.dim == 2
    classifier_fn=addSpectra.predict,
    labels=None,
    hide_color=1, # the spectrum is median normalized
    top_labels=1,
    # num_features=1000, # default= 100000
    num_samples=1_000,
    batch_size=10,
    segmentation_fn=segmentation_fn
    # distance_metric="cosine",
)

print(f"Finish explanation... Saving...", end="\n")

save_name = f"{name_galaxy}ExplanationUniform"

with open(f"{output_directory}/{save_name}.pkl", "wb") as file:

    pickle.dump(explanation, file)
###############################################################################
finish_time = time.time()
print(f"Run time: {finish_time-start_time:.2f}")
