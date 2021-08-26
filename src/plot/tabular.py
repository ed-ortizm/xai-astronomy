###############################################################################
class PlotExplanation:
    ############################################################################
    def plot_explanation(self, wave, spectrum, lime_array, vmin, vmax,
        s=3., linewidth=1., alpha=.7):

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(left=0.08, right=0.9)
        ########################################################################
        # line, = ax.plot(wave, spectrum[:-8], linewidth=linewidth, alpha=alpha)
        ########################################################################
        wave_explanation = wave[lime_array[:, 0].astype(np.int)]
        flux_explanation = spectrum[:-8][lime_array[:, 0].astype(np.int)]
        weights_explanation = lime_array[:, 1]
        ########################################################################
        #vmin = 0. #weights_explanation.min()
        #vmax = 1 #0.05*weights_explanation.max()
        ########################################################################
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        #norm = mpl.colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
        ########################################################################
        cmap = 'Reds' #'bwr'
        scatter = ax.scatter(wave_explanation, flux_explanation, s=s,
            c=weights_explanation, cmap=cmap, norm=norm,
            vmin=vmin, vmax=vmax, alpha=alpha, zorder=2.01)

        line, = ax.plot(wave, spectrum[:-8], c='black', linewidth=linewidth,
            alpha=alpha)

        ########################################################################
        ax_cb = self._colorbar_explanation(fig, vmin, vmax)
        ########################################################################
        spectrum_name = [f'{int(idx)}' for idx in spectrum[-8:-5]]
        spectrum_name = "-".join(spectrum_name)
        ########################################################################
        z = spectrum[-2]
        ########################################################################
        signal_noise_ratio = spectrum[-1]
        ########################################################################
        size='x-large'
        ax.set_title(f'spec-{spectrum_name} [SDSS-DR16]', fontsize=size)

        ax.set_xlabel('$\lambda$ $[\AA]$')
        ax.set_ylabel('Median normalized')

        ax.text(0.8, 0.9, f'z = {z:.4f}', transform=ax.transAxes, size=size)

        ax.text(0.8, 0.8, f'SNR = {signal_noise_ratio:.4f}',
            transform=ax.transAxes, size=size)
        # plt.tight_layout()
        ########################################################################
        return fig, ax, ax_cb, line, scatter
    ############################################################################
    def _colorbar_explanation(self, fig, vmin, vmax):
        # Make axes with dimensions as desired.
        ax_cb = fig.add_axes([0.91, 0.05, 0.03, 0.9])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        cmap = mpl.cm.Reds # -> mpl.cm.ScalarMappable
        #cmap = mpl.cm.bwr # -> mpl.cm.ScalarMappable
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # norm = mpl.colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,
            norm=norm, orientation='vertical', extend='both')

        cb.set_label('Lime weights')

        return cb
###############################################################################

###############################################################################
class OldPlotData:

    def __init__(self, spec, sdss_name, vmin, vmax):
        self.spec = spec
        self.sdss_name = sdss_name
        self.vmin = vmin
        self.vmax = vmax
        self._fig = None
        self._cmap = None

    def _colorbar_explanation(self):
        # Make axes with dimensions as desired.
        ax_cb = self._fig.add_axes([0.91, 0.05, 0.03, 0.9])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        self._cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=self._cmap,
                                        norm=norm,
                                        orientation='vertical', extend='both')
        cb.set_label('Lime weights')


        return cb

    def plot_explanation(self,
        wave_exp, flx_exp, weights_explanation,
        kernel_width, feature_selection, metric,
        s=3., linewidth=1., alpha=0.7):

        a = np.sort(weights_explanation)
        #print([f'{i:.2E}' for i in a[:2]])
        #print([f'{i:.2E}' for i in a[-2:]])

        self._fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(left=0.08, right=0.9)

        line, = ax.plot(self.spec, linewidth=linewidth, alpha=alpha)

        scatter = ax.scatter(wave_exp, flx_exp, s=s,
            c=weights_explanation, cmap='plasma',
            vmin=self.vmin, vmax=self.vmax, alpha=1.)

        ax_cb = self._colorbar_explanation()
        ax.set_title(
        f'{self.sdss_name}: {metric}, {feature_selection}, k_width={kernel_width}')

        # plt.tight_layout()

        return self._fig, ax, ax_cb, line, scatter
###############################################################################
