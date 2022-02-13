#! /usr/bin/env python3
################################################################################
from configparser import ConfigParser, ExtendedInterpolation
from functools import partial
import os
import sys
import time
################################################################################
from astropy.io import fits
####################################################################
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
####################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
################################################################################
from src.explainers.manga import manga
from src.explainers.manga import input_format
from src.plot import image
################################################################################
ti = time.time()
################################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('image.ini')
############################################################################
# Relevant data
plate_ifu = parser.get('data', 'plate_ifu')
manga_data_directory = parser.get('directories', 'manga')

flux = np.load(f'{manga_data_directory}/{plate_ifu}_image.npy')
####################################################################
# wave_master = np.linspace(3500, 7500, 4001)
model = manga.ToyModel(wave=None, cube=False)
explainer = lime_image.LimeImageExplainer()
####################################################################
# segmentation_fn = SegmentationAlgorithm('slic', chanel_axis=2)
####################################################################
explanation = explainer.explain_instance(
    flux,
    classifier_fn=model.predict,
    labels=(1,),
    hide_color=0,
    top_labels=1,
    # num_features=100_000,
    num_samples=1_000,
    batch_size=10,
    # segmentation_fn=segmentation_fn,
    # distance_metric='cosine',
    # model_regressor=None,
    random_seed=None
    )
####################################################################
image.explanation(
    flux,
    explanation,
    positive_only=True,
    number_features=5,
    hide_rest=False,
    show=True
    )
################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f} s')
################################################################################
