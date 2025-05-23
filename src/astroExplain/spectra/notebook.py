"""Utility functions to explore interpretability with jupyter notebooks"""

from functools import partial
import sys
from typing import Tuple

from lime.lime_image import ImageExplanation
# pylint: disable=C0411
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
import numpy as np

from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from autoencoders.ae import AutoEncoder
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.neighbors import SpectraNeighbors
from astroExplain.spectra.explanation import TellMeWhy
from astroExplain.spectra.explainer import LimeSpectraExplainer


def fig_axs_interpret_cluster(
    wave: np.array,
    mean_anomaly: np.array,
    median_anomaly: np.array,
    median_weights: np.array,
    mean_weights: np.array,
    fig_size=None,
) -> Tuple[Figure, Axes]:
    """
    Plot the mean and median anomaly and the mean and median
    explanation weights.

    INPUT

    wave: wavelength array
    mean_anomaly: mean anomaly array
    median_anomaly: median anomaly array
    median_weights: median explanation weights array
    mean_weights: mean explanation weights array
    fig_size: figure size

    OUTPUT

    fig: figure
    axs: axes
    """

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=fig_size,
        sharex=True,
        sharey=False,
        tight_layout=True,
    )

    axs[0].plot(wave, median_anomaly, c="black", label="Median")
    axs[0].set_ylabel("Normalized flux", fontsize=8)

    axs[1].plot(wave, mean_anomaly, c="black", label="Mean")
    axs[1].set_ylabel("Normalized flux", fontsize=8)

    axs[2].plot(wave, median_weights, color="black", label="Median")
    axs[2].plot(wave, mean_weights, color="blue", label="Mean")

    axs[2].set_ylabel("Explanation weight", fontsize=8)
    axs[2].set_ylim(0, 1.01)

    return fig, axs


def explanation_name(lime_config: dict, fudge_config: dict) -> str:
    """Retrieve the explanation name from the lime and fudge configs"""

    segmentation = lime_config["segmentation"]
    n_segments = lime_config["number_segments"]
    perturbation = fudge_config["kind_of_fudge"]

    explanation_str = f"{segmentation}_{n_segments}_{perturbation}"

    if perturbation == "scale":

        scale_factor = fudge_config["scale_factor"]
        explanation_str = f"{explanation_str}_{scale_factor}"

    elif perturbation == "flat":

        continuum = fudge_config["continuum"]
        explanation_str = f"{explanation_str}_{continuum}"

    elif perturbation == "gaussians":

        amplitude = fudge_config["amplitude"]
        explanation_str = f"{explanation_str}_{amplitude}"

    return explanation_str


def neighbors_explainer(
    number_samples: int,
    spectrum: np.array,
    segmentation: np.array,
    number_segments: int,
    fudge_parameters: dict,
) -> np.array:
    """
    Generate samples according to how LimeSpectraExplainer
    turns on and off segments and adds the corresponding
    fudge to it

    INPUT

    number_samples: number of lime samples with some segments
        replaced by the corresponding fudging
    spectrum: the spectrum from which to draw the lime samples
    segmentation: function to segments spectra, either kmeas or
        uniform segmentation
    number_segments: number of segments to divide input spectrum
    fudge_parameters: properties of fudging:
        fudge_parameters = {
        # same, same_shape, flat, with_mean, gaussians, scale
            "kind_of_fudge": "gaussians",
        # scale
            "scale_factor": 0.9,
        # flat
            "continuum": 1,
        # gaussians
            "amplitude":0.1,
            "sigmas_in_segment": 8,
        # control-noise
            "same_noise": True,
            "kernel_size": 3,
            "sigma": 0
        }

    OUTPUT
    neighbors: array with lime samples. First entry is the
        original spectrum
    """

    neighborhood = SpectraNeighbors()
    segmenter = SpectraSegmentation()

    if segmentation == "uniform":

        segmentation_function = segmenter.uniform

    elif segmentation == "kmeans":

        segmentation_function = segmenter.kmeans

    else:

        print("Segmentation: {segmentation} is not defined")
        sys.exit()

    segmentation_function = partial(
        segmentation_function, number_segments=number_segments
    )

    neighbors = neighborhood.get_neighbors(
        number_samples=number_samples,
        fudge_parameters=fudge_parameters,
        spectrum=spectrum,
        segmentation_function=segmentation_function,
    )

    segments = segmentation_function(spectrum)

    if segmentation == "uniform":

        print(f"Number of segments: {number_segments}")

    elif segmentation == "kmeans":

        number_segments = np.unique(segments).size
        print(f"Number of segments: {number_segments}")

    return neighbors


def spectrum_in_segments(spectrum: np.array, segments: np.array):
    """
    Return array where each row contains fluxes values per
    segment. Row zero contains only the fluxes of segment
    zero and the rest of the fluxes are set to zero and
    so on

    OUTPUT

    fluxes_per_segment: array where each row contains
    fluxes of segment with the same row index and nans
    for the rest of fluxes

    """

    number_segments = np.unique(segments).size
    number_fluxes = spectrum.size

    fluxes_per_segment = np.empty((number_segments, number_fluxes))

    # substract 1 to match id to start at zero
    for segment_id in np.unique(segments - 1):
        flux = np.where(segments == segment_id, spectrum, np.nan)
        fluxes_per_segment[segment_id, :] = flux

    return fluxes_per_segment


def interpret(
    wave: np.array,
    explanation: ImageExplanation,
    figsize: tuple = (10, 5),
    positive: int = 5,
    negative: int = 5,
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
    fig, axs = why.plot_full_explanation(figure_size=figsize)

    _, positive_spectrum = why.positive_mask_and_segments(positive)
    _, negative_spectrum = why.negative_mask_and_segments(negative)
    weights_explanation = why.get_heatmap()

    axs[0].plot(why.wave, why.galaxy, c="black")
    axs[0].plot(why.wave, positive_spectrum, c="red")
    axs[0].plot(why.wave, negative_spectrum, c="blue")
    axs[0].set_ylabel("Normalized flux")

    max_weight = np.nanmax(np.abs(weights_explanation))
    # max_weight += 0.1*max_weight
    # axs[1].set_ylim(ymin=-max_weight, ymax=max_weight)

    axs[1].plot(
        why.wave, np.abs(weights_explanation) / max_weight, color="black"
    )
    # axs[1].plot(why.wave, weights_explanation)
    # axs[1].hlines(0, xmin=wave.min(), xmax=wave.max(), color="black")
    axs[1].set_ylabel("Explanation weight")
    # axs[1].set_xlabel("$\lambda$ [$\AA$]")

    return fig, axs


def explain_reconstruction_score(
    wave: np.array,
    spectrum: np.array,
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
            velocity_filter=score_config["velocity"],
        ),
        reconstruction_parameters=ReconstructionParameters(
            percentage=score_config["percentage"],
            relative=score_config["relative"],
            epsilon=score_config["epsilon"],
        ),
    )

    anomaly_score_function = partial(
        anomaly.score, metric=score_config["metric"]
    )
    # Set explainer instance
    print("Set explainer and Get explanations", end="\n")
    explainer = LimeSpectraExplainer(random_state=0)
    segmentation_fn = None
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
    (
        explanation,
        ret_exp_score,
        ret_exp_local_pred
    ) = explainer.explain_instance(
        spectrum=spectrum,
        classifier_fn=anomaly_score_function,
        segmentation_fn=segmentation_fn,
        fudge_parameters=fudge_config,
        explainer_parameters=lime_config,
    )

    return (
        explanation,
        ret_exp_score,
        ret_exp_local_pred
    )

def interpret_dual_axis(
    wave: np.array,
    explanation,
    figsize: tuple = (10, 5),
) -> tuple:
    """
    Visualize interpretability of anomaly scores using a dual y-axis plot.

    INPUT
    wave: wavelength grid
    explanation: output of LimeSpectraExplainer
    positive: number of segments to highlight in red in explanation
    negative: number of segments to highlight in blue in explanation

    OUTPUT
    fig, (ax1, ax2): ax1 is the flux axis, ax2 is the explanation axis
    """
    why = TellMeWhy(wave=wave, explanation=explanation)

    fig, ax1 = plt.subplots(figsize=figsize)

    weights_explanation = why.get_heatmap()

    # Plot on left y-axis: normalized flux
    ax1.plot(why.wave, why.galaxy, c="black", label="Flux")
    ax1.set_ylabel("Normalized flux", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Create right y-axis
    ax2 = ax1.twinx()
    max_weight = np.nanmax(np.abs(weights_explanation))
    ax2.plot(
        why.wave, np.abs(weights_explanation) / max_weight,
        color="darkgreen", label="Explanation Weight"
    )
    ax2.set_ylabel("Explanation weight", color="darkgreen")
    ax2.tick_params(axis="y", labelcolor="darkgreen")

    fig.tight_layout()
    return fig, (ax1, ax2)

def interpret_embedded_panel(
    wave: np.array,
    explanation,
    figsize: tuple = (10, 5)
) -> tuple:
    """
    Visualize interpretability of anomaly scores with an embedded
    explanation panel.

    INPUT
    wave: wavelength grid
    explanation: output of LimeSpectraExplainer

    OUTPUT
    fig, (ax_flux, ax_weight): matplotlib figure and axis objects
    """
    why = TellMeWhy(wave=wave, explanation=explanation)

    flux = why.galaxy
    weights = np.abs(why.get_heatmap())
    weights /= np.nanmax(weights)

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # Main panel for flux
    ax_flux = fig.add_subplot(gs[0])
    ax_flux.plot(why.wave, flux, color="black")
    ax_flux.set_ylabel("Normalized flux")

    # Embedded panel for explanation weights
    # pylint: disable=W1401
    ax_weight = fig.add_subplot(gs[1], sharex=ax_flux)
    ax_weight.plot(why.wave, weights, color="black")
    ax_weight.set_ylabel("Explanation weight")
    ax_weight.set_xlabel("$\lambda$ [$\AA$]")

    return fig, (ax_flux, ax_weight)
