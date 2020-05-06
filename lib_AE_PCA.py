import numpy as np
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
