"""Compute standard emission line ratios from galaxy spectra."""

import numpy as np
import pandas as pd

def compute_emission_line_ratios(fluxes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard emission line ratios from galaxy spectra.
    This function calculates various emission line ratios commonly used in
    astronomy to analyze the physical conditions of ionized gas in galaxies.
    The ratios computed include:
    - Balmer decrement (Hα/Hβ)
    - [NII]/Hα
    - [OIII]/Hβ
    - [OIII]/[OII]
    - O3N2 index (log10([OIII]/Hβ ÷ [NII]/Hα))
    - [SII]/Hα
    - [SII] density ratio ([SII]6717/[SII]6731)
    This function assumes that the input DataFrame contains the necessary flux
    columns for these calculations, specifically:
    - 'h_alpha_flux'
    - 'h_beta_flux'
    - 'nii_6584_flux'
    - 'oiii_5007_flux'
    - 'oii_3726_flux'
    - 'sii_6717_flux'
    - 'sii_6731_flux'
    
    Args:
        fluxes_df (pd.DataFrame): DataFrame containing fluxes
        of emission lines.

    Returns:
        pd.DataFrame: DataFrame with computed emission line ratios.
    """
    df = fluxes_df.copy()

    # Balmer decrement
    df["balmer_decrement"] = df["h_alpha_flux"].div(df["h_beta_flux"])

    # [NII]/Hα
    df["nii_to_halpha"] = df["nii_6584_flux"].div(df["h_alpha_flux"])

    # [OIII]/Hβ
    df["oiii_to_hbeta"] = df["oiii_5007_flux"].div(df["h_beta_flux"])

    # [OIII]/[OII]
    df["oiii_to_oii"] = df["oiii_5007_flux"].div(df["oii_3726_flux"])

    # O3N2 index: log10([OIII]/Hβ ÷ [NII]/Hα)
    numerator = df["oiii_5007_flux"].div(df["h_beta_flux"])
    denominator = df["nii_6584_flux"].div(df["h_alpha_flux"])
    ratio = numerator.div(denominator)
    df["o3n2_index"] = np.log10(ratio)

    # [SII]/Hα
    df["sii_to_halpha"] = (
        df["sii_6717_flux"] + df["sii_6731_flux"]
    ).div(df["h_alpha_flux"])

    # [SII] density ratio
    df["sii_density_ratio"] = df["sii_6717_flux"].div(df["sii_6731_flux"])

    # Replace inf/-inf with NaN to avoid issues
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
