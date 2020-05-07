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
pca = PCA(0.30)
tr_data= pca.fit_transform(mnist_data)
print(type(tr_data))
print('N° of componets: ',pca.n_components_)

# Let's see how efficient is the compression

inv_tr= pca.inverse_transform(tr_data)
##################################
plt.figure(figsize=(8,4));

# Original Image
plt.subplot(1, 2, 1);
plt.imshow(mnist_data[1].reshape(28,28), cmap = plt.cm.gray,\
interpolation='nearest', clim=(0, 255))
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20)

# 154 principal components
plt.subplot(1, 2, 2);
plt.imshow(inv_tr[1].reshape(28, 28),cmap = plt.cm.gray,\
interpolation='nearest', clim=(0, 255))
plt.xlabel('5 components', fontsize = 14)
plt.title('3% of Explained Variance', fontsize = 20)
plt.show()
plt.close()
##########################################33


t_f= time.time()

print(t_f - t_i)
