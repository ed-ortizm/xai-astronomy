import numpy as np
import matplotlib.pyplot as plt

def data_gen(m=500,w1=0.1,w2=0.3,noise=0.1):
    # 3D dataset
    np.random.seed(4)
    angles = np.random.rand(m)*3*np.pi/2 -0.5
    data = np.empty((m,3))
    data[:,0]= np.cos(angles) + np.sin(angles)/2 + noise*np.random.randn(m)/2
    data[:,1]= np.sin(angles)*0.7 + noise* np.random.randn(m)/2
    data[:,2]= data[:,0]*w1 + data[:,1]* w2 + noise*np.random.randn(m)
    return data
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
