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
    pcts = counts / len(metadata_df)

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

    @staticmethod
    def multi_dataset_bpt_plot(
        fig: plt.Figure,
        ax: Axes,
        datasets: list,
        xytext: tuple = (-5, 5),
        x_lim: list = [-1.5, 1.0],
        y_lim: list = [-1.5, 1.5],
    ):
        """
        Plots multiple datasets on a single BPT diagram with error bars.

        datasets: list of dicts. Format:
        [
            {
                "name": "Dataset Label",
                "color": "blue",
                "n2_ha_med": array, "n2_ha_p45": array, "n2_ha_p55": array,
                "o3_hb_med": array, "o3_hb_p45": array, "o3_hb_p55": array
            }, ...
        ]
        """
        # 1. Demarcation Lines
        (x_kauff, y_kauff), (x_kewley, y_kewley), (x_schaw, y_schaw) = (
            BPT.bpt_boundary_lines()
        )
        ax.plot(x_kauff, y_kauff, "k--", lw=2, label="Kauffmann+03")
        ax.plot(x_kewley, y_kewley, "k-", lw=2, label="Kewley+01")
        ax.plot(x_schaw, y_schaw, "k-.", lw=2, label="Schawinski+07")

        # 2. Plot each dataset
        for data in datasets:
            # Convert linear values to log space
            log_x_med = np.log10(data["n2_ha_med"])
            log_y_med = np.log10(data["o3_hb_med"])

            # Calculate asymmetric relative errors in log space
            xerr_lower = log_x_med - np.log10(data["n2_ha_p45"])
            xerr_upper = np.log10(data["n2_ha_p55"]) - log_x_med
            xerr = np.vstack((xerr_lower, xerr_upper))

            yerr_lower = log_y_med - np.log10(data["o3_hb_p45"])
            yerr_upper = np.log10(data["o3_hb_p55"]) - log_y_med
            yerr = np.vstack((yerr_lower, yerr_upper))

            # Scatter with error bars
            ax.errorbar(
                log_x_med,
                log_y_med,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color=data["color"],
                label=data["name"],
                capsize=3,
                alpha=0.8,
            )
            # 1. Retrieve true cluster IDs (fallback to index if not provided)
            cluster_ids = data.get("cluster_id", range(len(log_x_med)))

            # 2. Format labels using the true cluster ID
            labels = [f"({int(cid)})" for cid in cluster_ids]

            # 3. Retrieve the offsets dictionary (fallback to empty dict)
            offsets_dict = data.get("offsets", {})

            for x, y, label, cid in zip(log_x_med, log_y_med, labels, cluster_ids):
                if np.isnan(x) or np.isnan(y):
                    continue

                # Look up the custom offset by cluster ID, fallback to global xytext
                offset = offsets_dict.get(int(cid), xytext)

                ax.annotate(
                    label,
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    va="center",  # Center alignment makes coordinate math easier
                    ha="center",
                    color=data["color"],
                )
            # # Add labels (0), (1), etc.
            # labels = [f"({i})" for i in range(len(log_x_med))]

            # # Fetch custom offsets if provided, otherwise repeat default xytext
            # custom_offsets = data.get("offsets", [xytext] * len(labels))

            # for x, y, label, offset in zip(
            #     log_x_med, log_y_med, labels, custom_offsets
            # ):
            #     if np.isnan(x) or np.isnan(y):
            #         continue

            #     ax.annotate(
            #         label,
            #         (x, y),
            #         xytext=offset,  # <--- Uses granular offset
            #         textcoords="offset points",
            #         fontsize=9,
            #         fontweight="bold",
            #         va="center",  # Changed to center for easier coordinate math
            #         ha="center",  # Changed to center for easier coordinate math
            #         color=data["color"],
            #     )

        # 3. Formatting
        ax.text(-1.0, -0.5, "Star Forming", fontsize=14, ha="center")
        ax.text(-0.15, -0.5, "Composite", fontsize=14, ha="center", rotation=-70)
        ax.text(-0.5, 1.25, "Seyfert", fontsize=14, ha="center")
        ax.text(0.5, -0.5, "LINER", fontsize=14, ha="center")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r"$\log_{10}([\mathrm{N~II}] / \mathrm{H}\alpha)$", fontsize=14)
        ax.set_ylabel(r"$\log_{10}([\mathrm{O~III}] / \mathrm{H}\beta)$", fontsize=14)
        ax.tick_params(labelsize=12)
        # -------------------------------------------------------------
        # Split Legends
        # -------------------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()

        # The first 3 items are the boundary lines; the rest are the datasets
        line_handles, line_labels = handles[:3], labels[:3]
        data_handles, data_labels = handles[3:], labels[3:]

        # 1. Create and add the boundary lines legend in the lower left
        line_legend = ax.legend(
            line_handles, line_labels, loc="lower left", fontsize=8, frameon=False
        )
        ax.add_artist(line_legend)

        # 2. Create the dataset color code legend in the lower right
        ax.legend(
            data_handles, data_labels, loc="lower right", fontsize=8, frameon=False
        )

        return fig, ax

    @staticmethod
    def median_45_55_ratios(cluster_meta_df: pd.DataFrame) -> dict:
        """
        Groups by cluster ID, computes the 45th, 50th, and 55th percentiles
        for the BPT line ratios, and returns a formatted DataFrame.
        """
        cols = ["nii_to_halpha", "oiii_to_hbeta"]
        grouped = cluster_meta_df.groupby("cluster")[cols]

        p45 = grouped.quantile(0.45)
        p50 = grouped.quantile(0.50)
        p55 = grouped.quantile(0.55)

        return pd.DataFrame(
            {
                "cluster_id": p50.index.astype(int),
                "n2_ha_med": p50["nii_to_halpha"].values,
                "o3_hb_med": p50["oiii_to_hbeta"].values,
                "n2_ha_p45": p45["nii_to_halpha"].values,
                "n2_ha_p55": p55["nii_to_halpha"].values,
                "o3_hb_p45": p45["oiii_to_hbeta"].values,
                "o3_hb_p55": p55["oiii_to_hbeta"].values,
            }
        )


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
