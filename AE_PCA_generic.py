#!/usr/bin/env python3
from lib_AE_PCA import *
# 3D dataset
np.random.seed(4)
m= 500
w1, w2= 0.1, 0.3
noise= 0.1

angles = np.random.rand(m)*3*np.pi/2 -0.5
data = np.empty((m,3))
data[:,0]= np.cos(angles) + np.sin(angles)/2 + noise*np.random.randn(m)/2
data[:,1]= np.sin(angles)*0.7 + noise* np.random.randn(m)/2
data[:,2]= data[:,0]*w1 + data[:,1]* w2 + noise*np.random.randn(m)

plot_3D(data)
scaler= StandardScaler()

## PCA
X = scaler.fit_transform(data)

pca= PCA(n_components=2)
pca.fit(X)
sing_vals = pca.transform(X)
print(sing_vals.shape)
plot_2D(sing_vals)
#################


encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1))
history = autoencoder.fit(X, X, epochs=20)
latent = encoder.predict(X)
print(latent.shape)
plot_2D(latent)
plot_2D(np.array([latent[:,1],latent[:,0]]).T)
