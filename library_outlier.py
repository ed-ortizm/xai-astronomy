import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
################################################################################
class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """
    ############################################################################
    def __init__(self, metric:'str', model:'tf.keras.model'):
        """
        Init fucntion

        Args:

            metric: (str) the name of the metric used to compute the outlier score
                using the observed spectrum and its reconstruction. Possible

            p: (float > 0) in case the metric is the lp metric, p needs to be a
                non null possitive float [Aggarwal 2001]
        """

        self.metric = metric
        self.model = model
        # if metric=='lp' and p>0:
        #     self.p = p
        # if:
        #     print(f'For lp metric p must be positive, instead p={p}')
        #     print(f'Failed to instantiate class')
        #     sys.exit()
    ############################################################################
    def score(self, O, percentage, image=False):
        """
        Computes the outlier score according to the metric used to instantiate
        the class.

        Args:
            O: (2D np.array) with the original objects where index 0 indicates
            the object and index 1 the features of the object.

            R: (2D np.array) with the reconstructed objects where index 0
            indicates the object and index 1 the features of the object.

        Returns:
            A one dimensional numpy array with the outlier scores for objects
            present in O
        """
        if self.metric == 'mse':
            print(f'Computing the outlier scores')
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            #
            # print(O[0,0,:5,:], '\n')
            # print(O[:,0,:5,0], '\n')
            R = self.model.predict(O)
            # print(O.shape, R.shape)
            #
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            #
            # print(R[0,0,:5,:], '\n')
            # print(R[:,0,:5,0], '\n')
            # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            # print(O[:, 0,:, 0].shape, R[:, 0,:, 0].shape)
            if image:

                score = self._mse(O=O[:, 0,:, 0], R=R[:, 0,:, 0],
                    percentage=percentage)

                print(score.reshape(-1,1).shape)
                return score

            else:

                return self._mse(O=O, R=R, percentage=percentage)

        else:
            print(f'The provided metric: {self.metric} is not implemented yet')
            sys.exit()
    ############################################################################
    def _mse(self, O, R, percentage):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        mse = np.square(R - O)

        number_outlier_fluxes = int(percentage*mse.shape[1])
        highest_mse = np.argpartition(mse, -1 * number_outlier_fluxes,
            axis=1)[:, -1 * number_outlier_fluxes:]

        score = np.empty(highest_mse.shape)
        for n, idx in enumerate(highest_mse):

            score[n, :] = mse[n, idx]
        o_score = score.sum(axis=1)
        similarity =  o_score.max() - o_score
        o_similarity = np.empty((o_score.size, 2))
        o_similarity[:, 0] = o_score[:]
        o_similarity[:, 1] = similarity[:]

        print(o_similarity.shape, 'hhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        return o_similarity
        # outlier_scores = []
        # for percentage in percentages:
        #
        #     mse = np.square(R - O)
        #
        #     number_outlier_fluxes = int(percentage*mse.shape[1])
        #
        #     highest_mse = np.argpartition(
        #         mse, -1 * number_outlier_fluxes,
        #         axis=1)[:, -1 * number_outlier_fluxes:]
        #
        #     score = np.empty(highest_mse.shape)
        #
        #     for n, idx in enumerate(highest_mse):
        #
        #         score[n, :] = mse[n, idx]
        #
        #     outlier_scores.append(score.sum(axis=1))
        #
        # return outlier_scores
    ############################################################################
    def top_reconstructions(self, scores, n_top_spectra):
        """
        Selects the most normal and outlying objecs

        Args:
            scores: (1D np.array) outlier scores

            n_top_spectra: (int > 0) this parameter controls the number of
                objects identifiers to return for the top reconstruction,
                that is, the idices for the most oulying and the most normal
                objects.

        Returns:
            most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
                with the location indexes of the most normal and outlying
                object in the training (and pred) set.
        """

        spec_idxs = np.argpartition(scores,
            [n_top_spectra, -1 * n_top_spectra])

        most_normal_ids = spec_idxs[: n_top_spectra]
        most_oulying_ids = spec_idxs[-1 * n_top_spectra:]

        return most_normal_ids, most_oulying_ids
################################################################################
