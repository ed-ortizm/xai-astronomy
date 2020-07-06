#!/usr/bin/env python3
from lib_AE_PCA import *
# 3D data
n_points=500
w1=0.1
w2=0.3
noise=0.1
data= data_gen(n_points,w1,w2,noise)
#plot_3D(data)

pca= PCA(n_components=2)
pca.fit(data)
sing_vals = pca.transform(data)
print(sing_vals.shape)
print('The two principal components are: \n',pca.components_)
# Let's check if the two principal comoments are orthogonal
z1 = pca.components_[0]
z2 = pca.components_[1]
z12 = z1*z2
print('The dot product of the two compoments is: ',z12.sum())
print(np.sum(z1*z1))
print(pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())

#plot_2D(sing_vals)
#################
#
# encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
# decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
# autoencoder = keras.models.Sequential([encoder, decoder])
#
# autoencoder.compile(loss="mse")#, optimizer=keras.optimizers.SGD(lr=0.1))
# history = autoencoder.fit(X, X, epochs=20)
# latent = encoder.predict(X)
#
# print(latent.shape)
# plot_2D(latent)
# plot_2D(np.array([latent[:,1],latent[:,0]]).T)
