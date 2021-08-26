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
###############################################################################
class Explainer_parallel:

    def __init__(self, explainer_type, training_data, training_labels,
        feature_names, kernel_widths, features_selection,
        sample_around_instance, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=False,
        mode='regression', n_processes=None):
        # The input variable are lists

        self.k_widths = kernel_widths
        self.ftrs_slect = features_selection
        self.around_instance = sample_around_instance

        if n_processes == None:
            self.n_processes = mp.cpu_count()-1 or 1
        else:
            self.n_processes = n_processes


        # Fixed values

        self.xpl_type = explainer_type
        self.t_dat = training_data
        self.t_lbls = training_labels
        self.ftr_names = feature_names
        self.t_dat_stats = training_data_stats
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

        self.Ex_partial = partial(Explainer, explainer_type=self.xpl_type,
        training_data=self.t_dat, training_labels=self.t_lbls,
        feature_names=self.ftr_names, training_data_stats=self.t_dat_stats,
        discretize_continuous=self.discretize, discretizer=self.discretizer,
        verbose=self.verbose, mode=self.mode)

        self.explanations = None

    def get_explanations(self, x, regressor, sdss_name):

        params_grid = product([x], [regressor], [sdss_name],
            self.k_widths, self.ftrs_slect, self.around_instance)

        with mp.Pool(processes=self.n_processes) as pool:
            self.explanations = pool.starmap(self._get_explanation, params_grid)
            self._sizeof(self.explanations, itr_name='explanations')

        return self.explanations

    def _get_explanation(self, x, regressor, sdss_name,
        kernel_width, feature_selection, sample_around_instance):

        explainer = self.Ex_partial(kernel_width, feature_selection,
            sample_around_instance)

        self._sizeof(explainer, itr_name='explainer', is_itr=False)

        return [sdss_name, kernel_width, feature_selection, sample_around_instance,
            explainer.explanation(x, regressor)]

    def _sizeof(self, iterable, itr_name="iterable", is_itr=True):

        if is_itr:
            size = 0
            for itr in iterable:
                x = sys.getsizeof(itr)*1e-6
                print(f'The size of object from {itr_name} is: {x:.2f} Mbs')
                size += x
            print(f"The total size of {itr_name} is {size:.2f} Mbs")
        else:
            size =  sys.getsizeof(iterable)
            print(f"The total size of {itr_name} is {size:.2f} Mbs")
###############################################################################
class TabularExplainer:
    def __init__(self, kernel_width, feature_selection,
        sample_around_instance, explainer_type, training_data,
        training_labels, feature_names, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=True,
        mode='regression'):

        self.xpl_type = explainer_type
        self.tr_data = training_data
        self.tr_labels = training_labels
        self.ftr_names = feature_names
        self.k_width = kernel_width
        self.ftr_select = feature_selection
        self.tr_data_stats = training_data_stats
        self.sar_instance = sample_around_instance
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode


        self.explainer = self._tabular_explainer()
        x = sys.getsizeof(self.explainer)*1e-6
        print(f'The size of the explainer is: {x:.2f} Mbs')

    def _tabular_explainer(self):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.tr_data, training_labels=self.tr_labels,
            feature_names=self.ftr_names, kernel_width=self.k_width,
            feature_selection=self.ftr_select,
            training_data_stats=self.tr_data_stats,
            sample_around_instance=self.sar_instance,
            discretize_continuous=self.discretize, discretizer=self.discretizer,
            verbose = self.verbose, mode=self.mode)

        return explainer

    def explanation(self, x, regressor):


        xpl = self.explainer.explain_instance(x, regressor,
            num_features=x.shape[0])
        return xpl.as_list()
###############################################################################
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
    # print(f"Creating explainers")
    # # defining variables
    # ################################################################################
    # mode = parser.get('explainer', 'mode')
    # kernel_width = np.sqrt(train_data[:, :-8].shape[1])*0.75
    # # feature_selection: selects the features that have the highest
    # # product of absolute weight * original data point when
    # # learning with all the features
    # feature_selection = parser.get('explainer', 'feature_selection')
    # sample_around_instance = parser.get('explainer', 'sample_around')
    # feature_names = [i for i in range(train_data[:, :-8].shape[1])]
    ################################################################################
    # Gotta develop my class through inheritance
    # explainer = lime_tabular.LimeTabularExplainer(
    #             training_data=train_data[:, :-8],
    #             mode=mode,
    #             training_labels=scores,
    #             feature_names=feature_names,
    #             kernel_width=kernel_width,
    #             verbose=True,
    #             feature_selection=feature_selection,
    #             discretize_continuous=False,
    #             discretizer='quartile',
    #             sample_around_instance=True,
    #             training_data_stats=None)
