import ast
import sys
import lime
from lime import lime_tabular
import multiprocessing as mp
import numpy as np

###############################################################################
class SpectraTabularExplainer:
    """ Explanation"""

    def __init__(
        self,
        data: "np.array",
        parameters: "dictionary",
        anomaly_score_function: "regressor",
    ):

        """
        PARAMETERS

            data:
            parameter:
            anomaly_score function:
        """

        self.training_data = data

        [
            self.mode,
            self.kernel_width,
            self.kernel,
            self.verbose,
            self.feature_selection,
            self.sample_around_instance,
            self.random_state,
        ] = self._set_attributes_from_dictionary(parameters)

        self.regressor = anomaly_score_function

        self.explainer = None
        self._build_explainer()

    ###########################################################################
    def _set_attributes_from_dictionary(
        self, parameters: "dictionary"
    ) -> "list":
        """
        Sets attibutes of explainer from dictionary passed to constructor

        """

        mode = parameters["mode"]
        kernel_width = self.training_data.shape[1] * float(parameters["kernel_width"])

        if parameters["kernel"] == "None":
            kernel = None
        else:
            kernel = parameters["kernel"]

        verbose = ast.literal_eval(parameters["verbose"])
        feature_selection = parameters["feature_selection"]

        sample_around_instance = ast.literal_eval(
            parameters["sample_around_instance"]
        )

        random_state = ast.literal_eval(parameters["random_state"])

        return [
            mode,
            kernel_width,
            kernel,
            verbose,
            feature_selection,
            sample_around_instance,
            random_state,
        ]

    ###########################################################################
    def _build_explainer(self):

        self.explainer = self._tabular_explainer()

        x = sys.getsizeof(self.explainer) * 1e-6
        print(f"The size of the explainer is: {x:.2f} Mbs")

    ###########################################################################
    def _tabular_explainer(self):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            mode=self.mode,
            training_labels=None,
            feature_names=None,
            categorical_features=None,
            categorical_names=None,
            kernel_width=self.kernel_width,
            kernel=self.kernel,
            verbose=self.verbose,
            class_names=None,
            feature_selection=self.feature_selection,
            discretize_continuous=False,
            discretizer="quartile",
            sample_around_instance=self.sample_around_instance,
            random_state=self.random_state,
            training_data_stats=None,
        )

        return explainer

    ###########################################################################
    def explain_anomaly_score(
        self, spectrum: "numpy array", number_features: "int" = 0
    ) -> "list":

        if number_features == 0:
            number_features = spectrum.shape[0]

        explanation = self.explainer.explain_instance(
            spectrum, self.regressor, num_features=number_features
        )

        return explanation.as_list()

    ###########################################################################
    def explain_set_anomaly_score(
        self,
        spectra: "numpy array",
        number_features: "int" = 0,
        number_processes: "int" = 1,
    ) -> "list":

        # with mp.Pool(processes=number_processes) as pool:
        #     explanations = pool.map(explain_sngle, spectrum_list)
        pass
