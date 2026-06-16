import numpy as np
from functools import partial
from anomaly.latent import LatentIForestAnomalyScore
from astroExplain.spectra.segment import SpectraSegmentation
from astroExplain.spectra.explainer import LimeSpectraExplainer


def explain_iforest_score(
    spectrum: np.array,
    lime_config: dict,
    fudge_config: dict,
    encoder,
    iforest_model,
    fitted_scaler,
):
    """
    Generate LIME explanations for the Isolation Forest anomaly score.

    INPUTS
    spectrum: galaxy spectrum to explain (1D array).
    lime_config: explainer configuration dictionary.
    fudge_config: configuration for image fudging in explanation.
    encoder_function: Callable to encode raw spectra (e.g., ae_model.encode).
    iforest_model: Trained Isolation Forest model.
    fitted_scaler: StandardScaler previously fitted on the latent training set.

    OUTPUT
    explanation: ImageExplanation from lime.lime_image
    ret_exp_score: Float representing the LIME model's R^2 score.
    ret_exp_local_pred: Float representing the LIME local prediction.
    """

    # 1. Instantiate our custom score wrapper
    anomaly_pipeline = LatentIForestAnomalyScore(
        encoder_model=encoder,
        scaler_model=fitted_scaler,
        iforest_model=iforest_model,
    )

    # 2. Set explainer instance
    print("Set explainer and Get explanations", end="\n")
    explainer = LimeSpectraExplainer(random_state=0)

    # 3. Setup segmentation
    segmentation_fn = None
    if lime_config["segmentation"] == "kmeans":
        segmentation_fn = SpectraSegmentation().kmeans
    elif lime_config["segmentation"] == "uniform":
        segmentation_fn = SpectraSegmentation().uniform

    segmentation_fn = partial(
        segmentation_fn, number_segments=lime_config["number_segments"]
    )

    # 4. Ensure spectrum is exactly 2D before passing it to LIME
    if spectrum.ndim == 1:
        spectrum_2d = spectrum.reshape(1, -1)
    else:
        spectrum_2d = spectrum

    # 5. Get explanations passing the .score method as the classifier function
    explanation, ret_exp_score, ret_exp_local_pred = explainer.explain_instance(
        spectrum=spectrum_2d,
        classifier_fn=anomaly_pipeline.score,
        segmentation_fn=segmentation_fn,
        fudge_parameters=fudge_config,
        explainer_parameters=lime_config,
    )

    return explanation, ret_exp_score, ret_exp_local_pred
