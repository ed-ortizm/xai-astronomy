"""Compute standard emission line ratios from galaxy spectra."""

import numpy as np

class LineRatios:
    """
    A class to compute standard emission line ratios from galaxy spectra.

    Each method expects:
    - wave: 1D numpy array of wavelengths (in Å)
    - flux: 2D numpy array of shape (n_spectra, n_pixels) containing
        fluxes

    Line fluxes are estimated by summing over narrow wavelength windows
    centered on the emission lines.
    """

    def __init__(self, window: float = 5.0):
        """
        Initialize the LineRatios class.

        Parameters
        ----------
        window : float
            Half-width of the wavelength window (in Å) used to
            integrate each line.
        """
        self.window = window

    def _integrate_line(
        self, wave: np.ndarray, flux: np.ndarray, center: float
    ) -> np.ndarray:
        """
        Integrate the flux over a window centered at the given line.

        Parameters
        ----------
        wave : np.ndarray
            1D array of wavelengths.
        flux : np.ndarray
            2D array of fluxes with shape (n_spectra, n_pixels).
        center : float
            Central wavelength of the line.

        Returns
        -------
        np.ndarray
            1D array of integrated fluxes for each spectrum.
        """
        mask = (wave >= center - self.window) & (wave <= center + self.window)
        return np.trapz(flux[:, mask], wave[mask], axis=1)

    def balmer_decrement(
        self, wave: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Hα/Hβ Balmer decrement to estimate dust extinction.

        Parameters
        ----------
        wave : np.ndarray
            1D array of wavelengths.
        flux : np.ndarray
            2D array of fluxes.

        Returns
        -------
        np.ndarray
            Balmer decrement (Hα/Hβ) for each spectrum.
        """
        H_alpha = self._integrate_line(wave, flux, 6563)
        H_beta = self._integrate_line(wave, flux, 4861)
        return np.where(H_beta > 0, H_alpha / H_beta, np.nan)

    def nii_to_halpha(self, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Compute [N II] λ6584 / Hα ratio for BPT or metallicity diagnostics.

        Returns
        -------
        np.ndarray
            [N II]/Hα ratio for each spectrum.
        """
        N_ii = self._integrate_line(wave, flux, 6584)
        H_alpha = self._integrate_line(wave, flux, 6563)
        return np.where(H_alpha > 0, N_ii / H_alpha, np.nan)

    def sii_to_halpha(self, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Compute ([S II] λ6717 + λ6731) / Hα ratio for BPT diagnostics.

        Returns
        -------
        np.ndarray
            [S II]/Hα ratio for each spectrum.
        """
        S_ii_6717 = self._integrate_line(wave, flux, 6717)
        S_ii_6731 = self._integrate_line(wave, flux, 6731)
        H_alpha = self._integrate_line(wave, flux, 6563)
        return np.where(H_alpha > 0, (S_ii_6717 + S_ii_6731) / H_alpha, np.nan)

    def sii_density_ratio(
        self, wave: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """
        Compute [S II] λ6717 / λ6731 ratio to infer electron density.

        Returns
        -------
        np.ndarray
            [S II] density ratio for each spectrum.
        """
        S_ii_6717 = self._integrate_line(wave, flux, 6717)
        S_ii_6731 = self._integrate_line(wave, flux, 6731)
        return np.where(S_ii_6731 > 0, S_ii_6717 / S_ii_6731, np.nan)

    def oiii_to_hbeta(self, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Compute [O III] λ5007 / Hβ ratio for ionization diagnostics.

        Returns
        -------
        np.ndarray
            [O III]/Hβ ratio for each spectrum.
        """
        O_iii = self._integrate_line(wave, flux, 5007)
        H_beta = self._integrate_line(wave, flux, 4861)
        return np.where(H_beta > 0, O_iii / H_beta, np.nan)

    def oiii_to_oii(self, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Compute [O III] λ5007 / [O II] λ3727 ratio to estimate
        ionization parameter.

        Returns
        -------
        np.ndarray
            [O III]/[O II] ratio for each spectrum.
        """
        O_iii = self._integrate_line(wave, flux, 5007)
        O_ii = self._integrate_line(wave, flux, 3727)
        return np.where(O_ii > 0, O_iii / O_ii, np.nan)

    def o3n2_index(self, wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Compute O3N2 index = log10(([O III]/Hβ) / ([N II]/Hα)) for
        metallicity diagnostics.

        Returns
        -------
        np.ndarray
            O3N2 index for each spectrum.
        """
        O_iii = self._integrate_line(wave, flux, 5007)
        H_beta = self._integrate_line(wave, flux, 4861)
        N_ii = self._integrate_line(wave, flux, 6584)
        H_alpha = self._integrate_line(wave, flux, 6563)

        valid = (H_beta > 0) & (H_alpha > 0) & (N_ii > 0)
        ratio = np.full(O_iii.shape, np.nan)
        ratio[valid] = np.log10(
            (O_iii[valid] / H_beta[valid]) / (N_ii[valid] / H_alpha[valid])
        )
        return ratio
