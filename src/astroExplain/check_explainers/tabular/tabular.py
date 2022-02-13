class Explainer:
    def __init__(self,
        kernel_width,
        feature_selection,
        sample_around_instance,
        training_data,
        training_labels,
        feature_names,
        training_data_stats=None,
        discretize_continuous=False,
        discretizer='decile',
        verbose=True,
        mode='regression'):
        """

        INPUTS
        kernel_width:
        feature_selection:
        sample_around_instance:
        training_data:
        training_labels:
        feature_names:
        training_data_stats:
        discretize_continuous:
        discretizer:
        verbose:
        mode:

        """
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
        """ """
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
        """

        INPUTS
            regressor:

        OUTPUTS

        """
        xpl = self.explainer.explain_instance(x, regressor,
            num_features=x.shape[0])
        return xpl.as_list()
################################################################################
# class Explainer_parallel:
#
#     def __init__(self, explainer_type, training_data, training_labels,
#         feature_names, kernel_widths, features_selection,
#         sample_around_instance, training_data_stats=None,
#         discretize_continuous=False, discretizer='decile', verbose=False,
#         mode='regression', n_processes=None):
#         # The input variable are lists
#
#         self.k_widths = kernel_widths
#         self.ftrs_slect = features_selection
#         self.around_instance = sample_around_instance
#
#         if n_processes == None:
#             self.n_processes = mp.cpu_count()-1 or 1
#         else:
#             self.n_processes = n_processes
#
#
#         # Fixed values
#
#         self.xpl_type = explainer_type
#         self.t_dat = training_data
#         self.t_lbls = training_labels
#         self.ftr_names = feature_names
#         self.t_dat_stats = training_data_stats
#         self.discretize = discretize_continuous
#         self.discretizer = discretizer
#         self.verbose = verbose
#         self.mode = mode
#
#         self.Ex_partial = partial(Explainer, explainer_type=self.xpl_type,
#         training_data=self.t_dat, training_labels=self.t_lbls,
#         feature_names=self.ftr_names, training_data_stats=self.t_dat_stats,
#         discretize_continuous=self.discretize, discretizer=self.discretizer,
#         verbose=self.verbose, mode=self.mode)
#
#         self.explanations = None
#
#     def get_explanations(self, x, regressor, sdss_name):
#
#         params_grid = product([x], [regressor], [sdss_name],
#             self.k_widths, self.ftrs_slect, self.around_instance)
#
#         with mp.Pool(processes=self.n_processes) as pool:
#             self.explanations = pool.starmap(self._get_explanation, params_grid)
#             self._sizeof(self.explanations, itr_name='explanations')
#
#         return self.explanations
#
#     def _get_explanation(self, x, regressor, sdss_name,
#         kernel_width, feature_selection, sample_around_instance):
#
#         explainer = self.Ex_partial(kernel_width, feature_selection,
#             sample_around_instance)
#
#         self._sizeof(explainer, itr_name='explainer', is_itr=False)
#
#         return [sdss_name, kernel_width, feature_selection, sample_around_instance,
#             explainer.explanation(x, regressor)]
#
#     def _sizeof(self, iterable, itr_name="iterable", is_itr=True):
#
#         if is_itr:
#             size = 0
#             for itr in iterable:
#                 x = sys.getsizeof(itr)*1e-6
#                 print(f'The size of object from {itr_name} is: {x:.2f} Mbs')
#                 size += x
#             print(f"The total size of {itr_name} is {size:.2f} Mbs")
#         else:
#             size =  sys.getsizeof(iterable)
#             print(f"The total size of {itr_name} is {size:.2f} Mbs")
# ###############################################################################
