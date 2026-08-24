"""Compute standard emission line ratios from galaxy spectra."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def get_subclass_summary(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes value counts, percentages, and an appended Total row for SDSS subClass.
    """
    column: str = "subClass"
    if metadata_df.empty:
        return pd.DataFrame(columns=[column, "count", "pct"])

    # 1. Compute counts and percentages
    counts = metadata_df[column].value_counts(dropna=False)
    pcts = (counts / len(metadata_df)) * 100

    # 2. Build summary dataframe
    summary_df = pd.DataFrame(
        {
            column: counts.index.astype(str),
            "count": counts.values,
            "pct": pcts.values,
        }
    )

    # 3. Create and append the Total row
    total_row = pd.DataFrame(
        [
            {
                column: "Total",
                "count": summary_df["count"].sum(),
                "pct": summary_df["pct"].sum(),
            }
        ]
    )

    result_df = pd.concat([summary_df, total_row], ignore_index=True)
    return result_df


class BPT:
    """Class to compute and plot BPT diagrams."""

    @staticmethod
    def bpt_boundary_lines():
        """
        Returns the boundary lines for the BPT diagram based on the following
        references:
        - Kauffmann et al. (2003): Empirical Star-Forming line
        - Kewley et al. (2001): Theoretical Maximum Starburst line
        - Schawinski et al. (2007): Seyfert / LINER division
        Returns:
            tuple: A tuple containing three tuples, each representing the x and y
                coordinates of the boundary lines for the BPT diagram.
        """
        # Kauffmann et al. (2003) - Empirical Star-Forming line
        x_kauff = np.linspace(-2.0, 0.0, 1000)
        y_kauff = 0.61 / (x_kauff - 0.05) + 1.3

        # Kewley et al. (2001) - Theoretical Maximum Starburst line
        x_kewley = np.linspace(-2.0, 0.4, 1000)
        y_kewley = 0.61 / (x_kewley - 0.47) + 1.19

        # Schawinski et al. (2007) - Seyfert / LINER division
        x_schaw = np.linspace(-0.18, 1.5, 1000)
        y_schaw = 1.05 * x_schaw + 0.45

        return (x_kauff, y_kauff), (x_kewley, y_kewley), (x_schaw, y_schaw)

    @staticmethod
    def standar_bpt_plot(
        fig: plt.Figure,
        ax: Axes,
        n2_ha: np.ndarray,
        o3_hb: np.ndarray,
        labels: list,
        xytext: tuple = (0, 0),
        x_lim: list = [-1.5, 1.0],
        y_lim: list = [-1.5, 1.5],
    ):
        """
        Create a standard BPT diagram with demarcation lines and data points.
        Parameters:
            fig: matplotlib.figure.Figure
                Figure object to plot on.
            ax: matplotlib.axes.Axes
                Axes object to plot on.
            n2_ha: array-like
                log10([N II] 6584 / H_alpha) values.
            o3_hb: array-like
                log10([O III] 5007 / H_beta) values.
            labels: array-like
                Labels for the data points.
            xytext: tuple, optional
                Offset for the annotations (default is (0, 0)).
            x_lim: list, optional
                Limits for the x-axis (default is [-1.5, 1.0]).
            y_lim: list, optional
                Limits for the y-axis (default is [-1.5, 1.5]).
        """
        # Demarcation Lines
        (x_kauff, y_kauff), (x_kewley, y_kewley), (x_schaw, y_schaw) = (
            BPT.bpt_boundary_lines()
        )

        ax.plot(x_kauff, y_kauff, "k--", lw=2, label="Kauffmann+03 (SF/Composite)")

        ax.plot(x_kewley, y_kewley, "k-", lw=2, label="Kewley+01 (Composite/AGN)")

        ax.plot(x_schaw, y_schaw, "k-.", lw=2, label="Schawinski+07 (Seyfert/LINER)")

        # Data
        ax.scatter(n2_ha, o3_hb, color="red")
        # labels
        for x, y, label in zip(n2_ha, o3_hb, labels):

            if np.isnan(x) or np.isnan(y):
                continue

            ax.annotate(
                label,
                (x, y),
                # x points right and y points up from the marker
                xytext=xytext,
                # Use offset in points rather than axis coordinates
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
        # -------------------------------------------------------------
        # 3. Formatting
        # -------------------------------------------------------------
        # Add region text labels
        ax.text(-1.0, -0.5, "Star Forming", fontsize=14, ha="center")
        ax.text(-0.15, -0.5, "Composite", fontsize=14, ha="center", rotation=-70)
        ax.text(-0.5, 1.25, "Seyfert", fontsize=14, ha="center")
        ax.text(0.5, -0.5, "LINER", fontsize=14, ha="center")

        # Set axis limits and labels
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r"$\log_{10}([\mathrm{N~II}] / \mathrm{H}\alpha)$", fontsize=14)
        ax.set_ylabel(r"$\log_{10}([\mathrm{O~III}] / \mathrm{H}\beta)$", fontsize=14)
        ax.tick_params(labelsize=12)

        ax.legend(loc="lower left", fontsize=8, frameon=False)

        return fig, ax


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
    df["sii_to_halpha"] = (df["sii_6717_flux"] + df["sii_6731_flux"]).div(
        df["h_alpha_flux"]
    )

    # [SII] density ratio
    df["sii_density_ratio"] = df["sii_6717_flux"].div(df["sii_6731_flux"])
    cols_ratios = [
        "balmer_decrement",
        "nii_to_halpha",
        "oiii_to_hbeta",
        "oiii_to_oii",
        "o3n2_index",
        "sii_to_halpha",
        "sii_density_ratio",
    ]
    # Replace inf/-inf with NaN to avoid issues
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, cols_ratios
