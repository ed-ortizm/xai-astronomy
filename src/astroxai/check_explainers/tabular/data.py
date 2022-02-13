################################################################################
class ExplanationData:

    def __init__(self, explanation_file):

        self.explanation_file = explanation_file
        self.sdss_directory = "/home/edgar/Documents/pyhacks/interactive_plotting"
        # "/home/edgar/zorro/SDSSdata/data_proc"
        self.sdss_name = self.explanation_file.split('_')[0]
        # self.explanation_file.split('/')[-1].split('_')[0]
        self.spec = np.load(f'{self.sdss_directory}/{self.sdss_name}.npy')

    def get_explanation_data(self, n_line):

        explanation_dictionary = self.get_serialized_data()

        kernel_width = explanation_dictionary[f'{n_line}'][0]
        kernel_width = float(kernel_width)

        array_explanation = explanation_dictionary[f'{n_line}'][1]
        wave_explanation = array_explanation[:, 0].astype(np.int)
        flux_explanation = self.spec[wave_explanation]
        weights_explanation = array_explanation[:, 1]
        metric = explanation_dictionary[f'{n_line}'][3]
        feature_selection = explanation_dictionary[f'{n_line}'][4]

        return (wave_explanation,
                flux_explanation,
                weights_explanation,
                kernel_width, metric, feature_selection)

    def get_serialized_data(self):

         with open(f'{self.explanation_file}', 'rb') as file:
             return pickle.load(file)
################################################################################

class Explanation:

    def __init__(self, discretize_continuous=False):
        self.discretize_continuous = discretize_continuous

    def explanations_from_file(self, explanation_file_path: str, save=True):

        if not os.path.exists(explanation_file_path):

            print(f'There is no file {explanation_file_path}')
            return None

        sdss_name = explanation_file_path.split("/")[-1].split("_")[0]
        metric = explanation_file_path.split("/")[-1].split("_")[1].strip(".exp")
        explanation_dict = {}

        with open(explanation_file_path, "r") as file:

            explanation_lines = file.readlines()

            for idx, explanation_line in enumerate(explanation_lines):

                explanation_line = self._line_curation(explanation_line)

                k_width = explanation_line[1] # string
                feature_selection = explanation_line[2]
                sample_around_instance = explanation_line[3]

                explanation_array = self._fluxes_weights(
                    line=explanation_line[4:])

                explanation_dict[f'{idx}'] = [k_width, explanation_array,
                    sdss_name, metric,
                    f'{feature_selection}_{sample_around_instance}']

        return explanation_dict

    def _fluxes_weights(self, line):

        length = np.int(len(line)/2)
        fluxes_weights = np.empty((length,2))

        for idx, fw in enumerate(fluxes_weights):
            fw[0] = np.float(line[2*idx].strip("'flux "))
            fw[1] = np.float(line[2*idx+1])

        return fluxes_weights

    def _line_curation(self, line):
        for charater in "()[]'":
            line = line.replace(charater, "")
        return [element.strip(" \n") for element in line.split(",")]

    def plot_explanation(self, spec, wave_exp, flx_exp, weights_exp, s=10., linewidth=0.2, cmap='plasma_r', show=False, ipython=False):

        c = weights_exp/np.max(weights_exp)

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)

        fig.colorbar()

        fig.savefig(f'testing/test.png')
        fig.savefig(f'testing/test.pdf')
        if show:
            plt.show()
        if not ipython:
            plt.close()
###############################################################################
