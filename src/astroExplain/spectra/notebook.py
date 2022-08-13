"""Utility functions to explore interpretability with jupyter notebooks"""

from functools import partial

import numpy as np
from lime.lime_image import ImageExplanation

from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explanation import TellMeWhy
from astroExplain.spectra.explainer import LimeSpectraExplainer
from autoencoders.ae import AutoEncoder

def interpret(
    wave: np.array,
    explanation: ImageExplanation,
    positive:int=5, negative: int =5
) -> tuple:
    """
    Visualize interpretability of anomaly scores
    
    INPUT
    wave: wavelength grid
    explanation: output of LimeSpectraExplainer
    positive: number of segments to highlight in red in explanation
        visualization
    negative: number of segments to highlight in blue in explanation
        visualization

    OUTPUT

    fig, axs: fig and ax objects from plt.subplots
        fig, axs --> ncols = 1, nrows=2
    """

    why = TellMeWhy(wave=wave, explanation=explanation)
    fig, axs = why.plot_full_explanation()

    _, positive_spectrum = why.positive_mask_and_segments(positive)
    _, negative_spectrum = why.negative_mask_and_segments(negative)
    weights_explanation = why.get_heatmap()

    axs[0].plot(why.wave, why.galaxy, c="black")
    axs[0].plot(why.wave, positive_spectrum, c="red")
    axs[0].plot(why.wave, negative_spectrum, c="blue")

    max_weight = np.nanmax(np.abs(weights_explanation))
    max_weight += 0.1*max_weight
    axs[1].set_ylim(ymin=-max_weight, ymax=max_weight)

    axs[1].plot(why.wave, weights_explanation)
    axs[1].hlines(0, xmin=wave.min(), xmax=wave.max(), color="black")

    return fig, axs

def explain_reconstruction_score(
    wave:np.array, spectrum: np.array,
    score_config: dict,
    lime_config: dict,
    fudge_config: dict,
    model: AutoEncoder,
):

    """
    Generate explanations for the lp, mad and mse scores and its
    variations

    INPUTS
    wave: wavelength grid of galaxy to explain
    spectrum: galaxy to explain
    score_config: score to explain
    lime_config: explainer configuration
    fudge_config: configuration for image fudging in explanation
    model: API to trained auto encoder

    OUTPUT
    explanation: ImageExplanation from lime.lime_image
    """

    anomaly = ReconstructionAnomalyScore(
        # reconstruct_function
        model.reconstruct,
        filter_parameters=FilterParameters(
            wave=wave,
            lines=score_config["lines"],
            velocity_filter=score_config["velocity"]
        ),
        reconstruction_parameters=ReconstructionParameters(
            percentage=score_config["percentage"],
            relative=score_config["relative"],
            epsilon=score_config["epsilon"]
        )
    )

    anomaly_score_function = partial(
        anomaly.score, metric=score_config["metric"]
    )
    # Set explainer instance
    print(f"Set explainer and Get explanations", end="\n")
    explainer = LimeSpectraExplainer(random_state=0)

    if lime_config["segmentation"] == "kmeans":
        
        segmentation_fn = SpectraSegmentation().kmeans

    elif lime_config["segmentation"] == "uniform":

        segmentation_fn = SpectraSegmentation().uniform


    segmentation_fn = partial(
        segmentation_fn, number_segments=lime_config["number_segments"]
    )
    # Load galaxy
    # Compute anomaly score
    # convert spectrum to gray image
    spectrum = spectrum[np.newaxis, :]
    # Get explanations
    explanation = explainer.explain_instance(
        spectrum=spectrum,
        classifier_fn=anomaly_score_function,
        segmentation_fn=segmentation_fn,
        fudge_parameters = fudge_config,
        explainer_parameters = lime_config,
    )

    return explanation