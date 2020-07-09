import os
import urllib
from glob import glob

import numpy as np
import astropy.io.fits as pyfits
import multiprocessing as mp
import pandas as pd
import numpy as np
from scipy import interpolate
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

    pool = mp.Pool(processes=7)

    # http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
    f = partial(min_max_interp_i, gs, dbPath)
    res = pool.map(f, range(len(gs)))

    if (None, None, None) in res:
        print('removing none')
        res = list(set(res))
        print(f'len of res = {len(res)}')
        res.remove((None, None, None))

    min_max = np.array([(res[i][0], res[i][1]) for i in range(len(res))])
    min, max = np.min(min_max), np.max(min_max)

    # Master grid and interpolation
    m_wl_grid = np.linspace(min, max, 5_000)
    flxs = np.array([res[i][2](m_wl_grid) for i in range(len(res))])

    print('Job finished')

    return m_wl_grid, flxs

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
    min, max, flx_intp = min_max_interp(plate, mjd, fiberid, run2d, z, dbPath)

    return min, max, flx_intp

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
    dest = f'/{dir_path}/{fname}'


    if not(os.path.exists(dest)):
        print(f'File {dest} not found.')
        return None, None, None

    with pyfits.open(dest) as hdul:
        wl_rg = 10. ** (hdul[1].data['loglam'])
        flx = hdul[1].data['flux']

    # Discarding spectrum with more than 10% of indefininte valunes
    if (np.count_nonzero(~np.isfinite(flx)) > flx.size // 10 ):
        return None, None, None

    # Deredshifting & min & max
    z_factor = 1./(1. + z)
    wl_rg *= z_factor
    wl_min , wl_max = np.min(wl_rg), np.max(wl_rg)
    print(f'min = {wl_min:.2f} & max = {wl_max:.2f}')

    # Replacing indefinite values with the median
    flx[~np.isfinite(flx)] = np.nanmedian(flx)

    # Removing the median & Interpolating
    print(f'The median is {np.nanmedian(flx):.2f}')
    flx -= np.nanmedian(flx)

    # Interpolation function
    flx_intp = interpolate.interp1d(wl_rg, flx, fill_value='extrapolate')

    return wl_min, wl_max, flx_intp

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

    if __name__ == 'uRF_SDSS':      # Wrap Parallelized code for Windows OS
        print ('*** Getting  %d fits files ****' % (len(gs)))
        start = time()

        # Create fits' DB folder, if needed
        if not os.path.exists(dbPath):
            os.makedirs(dbPath)

        # Download .fits files parallelly
        pool = mp.Pool()
        f = partial(getFitFile_i, gs, dbPath)
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
        print(f'The run2d is {run2d}')
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
