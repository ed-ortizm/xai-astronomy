"""Functionality to generate plots for the publication"""

import matplotlib.pyplot as plt
import numpy as np

from astroExplain.spectra.explanation import TellMeWhy

def different_explanations(
    wave: np.array,
    explanations: list,
    figsize: tuple = None,
    labels = list[str]
) -> tuple:

    explanation_weights = []

    spectrum = TellMeWhy(
        wave=wave, explanation=explanations[0]
    ).galaxy

    for explanation in explanations:

        why = TellMeWhy(wave=wave, explanation=explanation)
        explanation = why.get_heatmap()

        explanation_weights.append(explanation)


    fig, axs = plt.subplots(
        nrows=1+len(explanation_weights), ncols=1,
        sharex=True, tight_layout=True, figsize=figsize
    )

    font_size = 7.
    axs[0].plot(wave, spectrum, c="black")
    axs[0].set_ylabel("Normalized\nflux", fontsize=font_size)

    for idx, weights in enumerate(explanation_weights):

        max_weight = np.nanmax(np.abs(weights))
        # max_weight += 0.1*max_weight
        # axs[1].set_ylim(ymin=-max_weight, ymax=max_weight)

        axs[idx+1].plot(
            why.wave, np.abs(weights)/max_weight, color="black",
            label=labels[idx]
        )
        # axs[1].plot(why.wave, weights_explanation)
        # axs[1].hlines(0, xmin=wave.min(), xmax=wave.max(), color="black")
        axs[idx+1].set_ylabel(
            "Explanation\nweight", fontsize=font_size    
    )

    axs[-1].set_xlabel("$\lambda$ [$\AA$]")


    return fig, axs