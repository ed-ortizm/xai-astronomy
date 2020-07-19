import os
import urllib
from glob import glob
from time import time, sleep

import numpy as np
import astropy.io.fits as pyfits
import multiprocessing as mp
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from functools import partial
import matplotlib
import matplotlib.pyplot as plt

## Me

def spectra(gs, dbPath):
    """
    Computes the spectra interpolating over a master grid of wavelengths
    Parameters
    ----------
    gs : Pandas DataFrame with info of the galaxies
    dbPath : String : Path to data base

    Returns
    -------
    m_wl_grid : numpy array 1-D : The master wavelength grid
    flxs :  numpy array 2-D : Interpolated spectra over the grid
    """

    print(f'Getting grid of wavelengths and spectra from {len(gs)} .fits files')
    # http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    f = partial(min_max_interp_i, gs, dbPath)

    # close the pool (with) & do partial before
    with mp.Pool() as pool:
        res = pool.map(f, range(len(gs)))

    if ('No value', 'No value') in res:
        ##  Removing None from res
        res = list(filter(lambda a: a != ('No value', 'No value'), res))

    ## wavelenght arrays
    # sizes of each array

    sizes = np.array([r[0].size for r in res])
    wls = np.empty((sizes.size, np.max(sizes)))

    for idx, r in enumerate(res):
        wls[idx] = np.pad(r[0], pad_width=(0, wls.shape[1]-r[0].size), constant_values=np.nan)

    min, max = np.min(wls), np.max(wls)

    # Master grid
    wl_grid = np.linspace(min, max, 5_000)

    # Discarding spectrum with more than 10% of indefininte
    # valunes in a given wl for al training set

    flxs = np.empty(wls.shape)
    for idx, r in enumerate(res):
        flxs[idx] = np.pad(r[1], pad_width=(0,flxs.shape[1]-r[1].size), constant_values=np.nan)

    wkeep = np.where(np.count_nonzero(~np.isfinite(flxs), axis=0) < flxs.shape[0] / 10)

    # Removing one dimensional axis since wkeep is a tuple
    print(flxs.shape, wls.shape)
    flxs = np.squeeze(flxs[:, wkeep])
    wls = np.squeeze(wls[:, wkeep])
    print(flxs.shape, wls.shape)

    # Replacing indefinite values in a spectrum with its nan median
    for flx in flxs:
        flx[np.where(~np.isfinite(flx))] = np.nanmedian(flx)

    # Normalizing by the median
    flxs -= np.median(flxs, axis=1).reshape((flxs.shape[0],1))

    # Interpolating
    print(flxs.shape, wls.shape)
    flxs = interp1d(wls, flxs, axis=1)(wl_grid)

    print('Job finished')

    return wl_grid, flxs

def min_max_interp_i(gs, dbPath, i):
    """
    Computes the min and max value in the wavelenght grid for the ith spectrum.
    Computes the interpolation functtion for the ith spectrum.
    Parameters
    ----------
    gs : Pandas DataFrame with info of the galaxies
    dbPath : String : Path to data base
    i : int : Index of .fits file in the DataFrame
    Returns
    -------
    min : float : minimum value if the wavelength grid for the ith spectrum
    max : float : maximun value if the wavelength grid for the ith spectrum
    flx_intp :  interp1d object : Interpolation function for the ith spectrum
    """

    obj = gs.iloc[i]
    plate = obj['plate']
    mjd = obj['mjd']
    fiberid = obj['fiberid']
    run2d = obj['run2d']
    z = obj['z']
    wl_rg, flx = min_max_interp(plate, mjd, fiberid, run2d, z, dbPath)

    return wl_rg, flx

def min_max_interp(plate, mjd, fiberid, run2d, z, dbPath):
    """
    Computes the min and max value in the wavelenght grid for the spectrum.
    Computes the interpolation functtion for the spectrum.
    Parameters
    ----------
    plate : int : plate number
    mjd : int : mjd of observation (days)
    fiberid : int : Fiber ID
    run2d : str : 2D Reduction version of spectrum
    z : float : redshift, replaced by z_noqso when available.
    (z_nqso --> Best redshift when excluding QSO fit in BOSS spectra (right redshift to use for galaxy targets))
    dbPath : String : Path to data base
    Returns
    -------
    wl_min : float : minimum value of the wavelength grid for the spectrum
    wl_max : float : maximun value of the wavelength grid for the spectrum
    flx_intp :  interp1d object : Interpolation function for the spectrum

    """
    # Path to the .fits file of the target spectrum
    fname = f'spec-{plate:04}-{mjd}-{fiberid:04}.fits'
    SDSSpath = f'/sas/dr16/sdss/spectro/redux/{run2d}/spectra/lite/{plate:04}/'
    dir_path = f'/{dbPath}/{SDSSpath}'
    dest = f'{dir_path}/{fname}'


    if not(os.path.exists(dest)):
        print(f'File {dest} not found.')
        return 'No value', 'No value'

    with pyfits.open(dest) as hdul:
        wl_rg = 10. ** (hdul[1].data['loglam'])
        flx = hdul[1].data['flux']


    # Deredshifting & min & max
    z_factor = 1./(1. + z)
    wl_rg *= z_factor

    return wl_rg, flx

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

## From Guy Goren, Dovi's student

def getFitsFiles(gs, dbPath):

    """
    Downloads .fits files from SDSS dr16 for the galaxies specified in the gs DataFrame to dbPath (parallel version)
    :param gs: Pandas Dataframe: Containing the fields 'plate', 'mjd', 'fiberid', run2d'
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :param fitsFolder: String: fitsFolder name inside dbPath to place the .fits file
    :return: None.
    """

    print ('*** Getting  %d fits files ****' % (len(gs)))
    start = time()

    # Create fits' DB folder, if needed
    if not os.path.exists(dbPath):
        os.makedirs(dbPath)

    # Download .fits files parallelly
    pool = mp.Pool()
    f = partial(getFitFile_i, gs, dbPath)
    print('Starting first pool')
    res = pool.map(f, range(len(gs)))
    nFailed=sum(res)

    end = time()
    print('  Done! Finished downloading .fits files\t')
    print(f'  Failed to download: {nFailed} of {len(gs.index)}')
    print('  Time: ' + str(round(end - start, 2)) + ' [s]')


def getFitFile_i(gs, dbPath, i):
    """
    Downloading the ith .fits file in the gs DataFrame to dbPath.
    :param gs: Pandas Dataframe: Containing the fields 'plate', 'mjd', 'fiberid', run2d'
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :param i: Int: The index of the galaxy's .fits file to download
    :return: 0, if succeeded to download. 1 if failed.
    """

    # Extract identifiers
    obj = gs.iloc[i]
    plate = str(obj['plate']).zfill(4)
    mjd = str(obj['mjd'])
    fiberid = str(obj['fiberid']).zfill(4)
    run2d = str(obj['run2d'])

    # Try & Except a failed Download
    try:
        getFitsFile(plate, mjd, fiberid, run2d, dbPath)
        return 0
    except Exception as e:
        print ('Failed to DL: ' + 'https://data.sdss.org/' + 'sas/dr16/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/' + '-'.join(['spec',plate,mjd,fiberid]) + '.fits')
        print(str(e))
        return 1


def getFitsFile(plate, mjd, fiberid, run2d, dbPath):

    """
    Download a .fits files by plate,mjd,fiberid & run2d to dBPath
    :param plate: String (4 digits): Plate number
    :param mjd: String: mjd
    :param fiberid: String (4 digits): fiber ID
    :param run2d: String: run2d
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :return: None.
    """

    # Calculate paths
    filename = '-'.join(['spec',plate,mjd,fiberid]) + '.fits'
    SDSSpath = 'sas/dr16/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/'
    url = 'https://data.sdss.org/' + SDSSpath + filename
    folderPath = dbPath + SDSSpath
    dest = folderPath + filename

    # Create folder for the .fits file, if needed
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    # Try to download the .fits file 10 times
    # A file under 60kB is considered a failure
    if not(os.path.isfile(dest)):
        urllib.request.urlretrieve(url, dest)

    j = 0
    while j < 10 and (os.path.getsize(dest) < 60000):
        os.remove(dest)
        urllib.request.urlretrieve(url, dest)
        j += 1
        sleep(1)
    if (os.path.getsize(dest) < 60000):
        print('Size is: ' + str(os.path.getsize(dest)) + ' '),
        os.remove(dest)
        raise Exception('Spectra wasn\'t found')
