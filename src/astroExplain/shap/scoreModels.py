"""
Code to adapt variational autoencoder for SHAP explanations with DeepExplainer
VAE reconstructs spectra, this module takes a pre-trained VAE and uses it to
build a score model for SHAP explanations.
"""

import tensorflow as tf
from keras import ops

def build_mse_score_model(
    in_tensor: tf.Tensor,
    out_tensor: tf.Tensor,
    print_summary: bool = False,
)-> tf.keras.Model:
    """
    Builds a Keras model that computes the mean squared error between input
    and output tensors.
    Args:
        in_tensor (tf.Tensor): Input tensor, typically the input to the VAE.
        out_tensor (tf.Tensor): Output tensor, typically the reconstructed output
            from the VAE.
        print_summary (bool): If True, prints the model summary.
    Returns:
        tf.keras.Model: A Keras model that computes the mean squared error.
    """

    squared_error = ops.square(in_tensor - out_tensor)
    score = ops.mean(squared_error, axis=-1)
    print(f"Shape of the score tensor: {score.shape}")
    score = ops.expand_dims(score, axis=-1)
    print(f"Shape of the score tensor after expansion: {score.shape}")

    score_model = tf.keras.Model(inputs=in_tensor, outputs=score)

    if print_summary:
        print("\nAnomaly score model created successfully:")
        score_model.summary()

    return score_model
