import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
def data_gen():
    pass
def plot_2D(data):
    fig = plt.figure()
    plt.plot(data[:,0], data[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    #save_fig("linear_autoencoder_pca_plot")
    plt.show()
    plt.close()

def plot_3D(data):
    x,y,z = data[:,0], data[:,1], data[:,2]
    fig = plt.figure()
    ax= plt.axes(projection='3d')
    ax.scatter3D(x,y,z)
    plt.show()
    plt.close()

def implot_pca(original,reconstructed,components):
    plt.figure(figsize=(8,4));

    # Original Image
    plt.subplot(1, 2, 1);
    plt.imshow(original.reshape(28,28), cmap = plt.cm.gray,\
    interpolation='nearest', clim=(0, 255))
    plt.xlabel('784 components', fontsize = 14)
    plt.title('Original Image', fontsize = 20)

    # principal components
    plt.subplot(1, 2, 2);
    plt.imshow(reconstructed.reshape(28, 28),cmap = plt.cm.gray,\
    interpolation='nearest', clim=(0, 255))
    plt.xlabel(str(components) + 'components', fontsize = 14)
    plt.title('Reconstructed image', fontsize = 20)
    plt.show()
    plt.close()
