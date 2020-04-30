#!/usr/bin/env python3
#from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# encoder=\
# keras.models.Sequential([keras.layers.Dense(2,input_shape=[3])])
# decoder=\
# keras.models.Sequential([keras.layers.Dense(3,input_shape=[2])])
# autoencoder=\
# keras.models.Sequential([encoder,decoder])
#
# autoencoder.compile(loss="mse")
# optimizer=keras.optimizers.SGD(lr=0.1)
x = np.linspace(-1,1,100)
X,Y= np.meshgrid(x, x)
z =  X**2+Y**2
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(X,z,'blue')
plt.show()
plt.close()
