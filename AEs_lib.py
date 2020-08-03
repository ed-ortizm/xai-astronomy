from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from constants_AEs import data_proc

class Outlier:

    def __init__(self, dbPath=data_proc):
        self.scores = None
        self.fnames = glob(f'{dbPath}/*')

    def xi(self, O, P):
        self.scores, p = chisquare(O,P, axis=1)
        self.scores = np.abs(self.scores)
        return self.scores

    def mse(self, O, P):
        self.scores = np.square(O-P).mean(axis=1)
        return self.scores


    def mad(self, O, P):
        self.scores = np.abs(O-P).mean(axis=1)
        return self.scores

    def retrieve(self, scores=None, N=20, metric=None):
        print('Loading outlier scores!')

        ids_proc_specs = np.argpartition(scores, -1*N)[-1*N:]

        print('Outlier scores for the weirdest spectra')

        file = open(f'{metric}_oulier_scores.dat', 'w+')

        for n, idx in enumerate(ids_proc_specs):
            print(f'ID:{idx} --> {scores[idx]}. File name: {self.fnames[idx]}')
            file.write(f'ID:{idx} --> {scores[idx]}. File name: {self.fnames[idx].split("/")[-1]}\n')

        file.close()

class AEpca:

    def __init__(self, in_dim, lat_dim=2, batch_size=32, epochs=10, lr= 1e-4):
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.lat_dim = lat_dim
        self.epochs = epochs
        self.lr = lr
        self.encoder = None
        self.decoder = None
        self.AE = None
        self._init_AE()

    def _init_AE(self):

        # Build Encoder
        inputs = Input(shape=(self.in_dim,), name='encoder_input')
        latent = Dense(self.lat_dim, name='latent_vector')(inputs)
        self.encoder = Model(inputs, latent, name='encoder')
        self.encoder.summary()
#        plot_model(self.encoder, to_file='encoder.png', show_shapes='True')

        # Build Decoder
        latent_in = Input(shape=(self.lat_dim,), name='decoder_input')
        outputs = Dense(self.in_dim, name='decoder_output')(latent_in)
        self.decoder = Model(latent_in, outputs, name='decoder')
        self.decoder.summary()
#        plot_model(self.decoder, to_file='decoder.png', show_shapes='True')

        # AE = Encoder + Decoder
        autoencoder = Model(inputs, self.decoder(self.encoder
                            (inputs)), name='autoencoder')
        autoencoder.summary()
#        plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

        # Mean square error loss function with Adam optimizer
        autoencoder.compile(loss='mse', optimizer='adam') #, lr = self.lr)

        self.AE = autoencoder

    def fit(self, spectra):
        self.AE.fit(spectra, spectra, epochs=self.epochs,
                    batch_size=self.batch_size)

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)
        return self.AE.predict(test_spec)

    def encode(self, spec):
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        return self.encoder(spec)

    def decode(self, lat_val):
        return self.decoder(lat_val)

    def save(self):
        self.encoder.save('encoder')
        self.decoder.save('decoder')
        self.AE.save('AutoEncoder')

class PcA:

    def __init__(self, n_comps = False):
        if not(n_comps):
            self.PCA = PCA()
        else:
            self.n_comps = n_comps
            self.PCA = PCA(self.n_comps)

    def fit(self, spec):
        return self.PCA.fit(spec)

    def components(self):
        return self.PCA.components_

    def expvar(self):
        return self.PCA.explained_variance_ratio_

    def inverse(self, trf_spec):
        if trf_spec.ndim == 1:
            trf_spec = trf_spec.reshape(1, -1)

        return self.PCA.inverse_transform(trf_spec)

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)

        return self.PCA.transform(test_spec)

def plt_spec_pca(flx,pca_flx,componets):
    '''Comparative plot to see how efficient is the PCA compression'''
    plt.figure(figsize=(8,4));

    # Original Image
    plt.subplot(1, 2, 1);
    plt.plot(flx)
    plt.xlabel(f'{flx.size} components', fontsize = 14)
    plt.title('Original Spectra', fontsize = 20)

    # principal components
    plt.subplot(1, 2, 2);
    plt.plot(pca_flx)
    plt.xlabel(f'{componets} componets', fontsize = 14)
    plt.title('Reconstructed spectra', fontsize = 20)
    plt.show()
    plt.close()

def plot_2D(data, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(data[:,0], data[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.savefig(f'{title}.png')
    plt.show()
    plt.close()

# ## Checking the percentages of explained variance
#
# tot_var = sum(pca.explained_variance_)
#
# expl_var = [(i/tot_var)*100 for i in sorted(pca.explained_variance_, reverse=True)]

# n = 10
# for i in range(n):
#     print(f'Component N° {i} explains {expl_var[i]:.3}% of the vatiance')
#
# print(f'These first {n} components explain {sum(expl_var[:n]):.6}% of the variance')

#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-4,
#    decay_steps=10000,
#    decay_rate=0.9)
##optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
#autoencoder.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
