#!/usr/bin/env python3
from lib_AE_PCA import *
import time
import os.path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# The MNIST database of handwritten digits has 784 feature columns
# (784 dimensions), a training set of 60,000 examples, and a test set
# of 10,000 examples.
t_i= time.time()
## Download and load the data
if os.path.exists('mnist_data.npy') and os.path.exists('mnist_labels.npy'):
    mnist_data= np.load('mnist_data.npy',allow_pickle=True)
    mnist_labels= np.load('mnist_labels.npy',allow_pickle=True)
else:
    mnist = fetch_openml('mnist_784')
    np.save('mnist_data',mnist.data)
    np.save('mnist_labels',mnist.target)

## Standarize the dataset
# --> We need this beacuse PCA is affected by scale (variance)
# We fit on the training set and transform on the training and test set
scaler= StandardScaler()
scaler.fit_transform(mnist_data)


# Now we fit PCA on the training set
# pca = PCA(0.95)
# tr_data= pca.fit_transform(mnist_data)
# print(type(tr_data))
# print('N° of componets: ',pca.n_components_)
#
# # Let's see how efficient is the compression
#
# inv_tr= pca.inverse_transform(tr_data)
# implot_pca(mnist_data[0],inv_tr[0],pca.n_components_)
# images
# Now let's try the same with AEs
# the loss function was giving nan, I read this
# https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training

encoder = keras.models.Sequential([keras.layers.Dense(154, input_shape=[784])])
decoder = keras.models.Sequential([keras.layers.Dense(784, input_shape=[154])])
autoencoder = keras.models.Sequential([encoder, decoder])

# I'm getting a bad loss function
# https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training
autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-10))


if not(os.path.exists('autoencoder.json')):
    history = autoencoder.fit(mnist_data, mnist_data, epochs=20)
    model_encoder= encoder.to_json()
    with open('encoder.json','w') as json_file:
        json_file.write(model_encoder)
    encoder.save_weights('encoder.h5')

    model_decoder= decoder.to_json()
    with open('decoder.json','w') as json_file:
        json_file.write(model_decoder)
    decoder.save_weights('decoder.h5')

    model_autoencoder= autoencoder.to_json()
    with open('autoencoder.json','w') as json_file:
        json_file.write(model_autoencoder)
    encoder.save_weights('autoencoder.h5')
else:
    ## Load the models

    # encoder
    json_file= open('encoder.json','r')
    loaded_model_json= json_file.read()
    json_file.close()
    encoder= keras.models.model_from_json(loaded_model_json)
    # decoder
    json_file= open('decoder.json','r')
    loaded_model_json= json_file.read()
    json_file.close()
    decoder= keras.models.model_from_json(loaded_model_json)
    # Encoder
    json_file= open('autoencoder.json','r')
    loaded_model_json= json_file.read()
    json_file.close()
    autoencoder= keras.models.model_from_json(loaded_model_json)

reconstruction = autoencoder.predict(mnist_data)
implot_pca(mnist_data[0],reconstruction[0],154)
# print(mnist_data[0].shape)

# Saving the model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

t_f= time.time()

print(t_f - t_i)
