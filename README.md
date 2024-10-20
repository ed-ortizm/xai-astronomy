# Interpretable Machine Learning for Outlier Detection in SDSS Galaxy Spectra

## Overview

This repository provides **interpretable machine learning (IML)** tools for explaining a pre-trained Variational Autoencoder (VAE) model used for outlier detection in galaxy spectra from the Sloan Digital Sky Survey (SDSS). It leverages **LIME (Local Interpretable Model-agnostic Explanations)** to help visualize and interpret why certain spectra are flagged as anomalies. This repository assumes that the model is already trained and focuses on the interpretability aspect of machine learning.

## Features

- **LIME for Model Interpretability**: Integration of LIME to explain the VAE outlier predictions on galaxy spectra.
- **Visualization of Anomalies**: Tools to visualize which features in the spectra contribute to their classification as anomalies.
- **Image and Spectra Explanation Modules**: Separate modules for explaining model predictions for both image data and galaxy spectra.

## Getting Started

### Prerequisites

- Python
- LIME
- A pre-trained model for outlier detection (not included in this repository)
- SDSS Galaxy Spectra dataset

### Installation

Clone the repository:

```bash
git clone https://github.com/ed-ortizm/xai-astronomy
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare the Pre-trained Model**: Ensure you have a pre-trained model for outlier detection on galaxy spectra from SDSS.
2. **Run LIME for Interpretability**: Use the provided scripts to apply LIME to the model's predictions. LIME will generate explanations for why specific spectra were classified as outliers.
3. **Visualize Explanations**: The repository includes tools to visualize the explanations, making it easier to interpret which spectral features are driving the anomaly classification.

### Repository Structure

```bash
├── data                     # Data used for exploration
├── image                    # Image explanation module
│   └── [image explanation scripts and notebooks]
├── iml_use_cases            # Example use cases for IML, such as clustering and score comparison
│   └── [use cases notebooks]
├── spectra                  # Spectra explanation module
│   └── [spectra explanation scripts and notebooks]
├── src                      # Source code for explanation modules
│   ├── image_explanation    # Contains scripts for image explanation
│   ├── spectra_explanation  # Contains scripts for spectra explanation
├── LICENSE                  # License information
├── README.md                # This README file
├── requirements.txt         # Python dependencies
├── setup.py                 # Setup script
```

## Contact

For any questions or suggestions, feel free to reach out to:

- Edgar Ortiz (ed.ortizm@gmail.com)
- Mederic Boquien (mederic.boquien@oca.eu)

We welcome contributions and feedback to improve this repository!