import os
import urllib
from glob import glob

import numpy as np
import astropy.io.fits as pyfits
import multiprocessing as mp
import pandas as pd
import numpy as np

## Me

def min_max_wl(plate, mjd, fiberid, run2d, z, dbPath):
    """Rturns the minimun and maximun value the wavelength range"""

    # Path file for the target spectrum
    fname = f'spec-{plate:04}-{mjd}-{fiberid:04}.fits'
    SDSSpath = f'/sas/dr16/sdss/spectro/redux/{run2d}/spectra/lite/{plate:04}/'
    dir_path = f'/{dbPath}/{SDSSpath}'
    dest = f'/{dir_path}/{fname}'

    # Deredshifting
    with pyfits.open(dest) as hdul:
        n_pixels = hdul[1].header['NAXIS2']
        COEFF0 = hdul[0].header['COEFF0']
        COEFF1 = hdul[0].header['COEFF1']
        flx = hdul[1].data['flux']

    wl_rg = 10. ** (COEFF0 + COEFF1 * np.arange(n_pixels))
    z_factor = 1./(1. + z)
    wl_rg *= z_factor
    wl_min , wl_max = np.min(wl_rg), np.max(wl_rg)

    # Removing median & Interpolating
    flx -= np.nanmedian(flx)
    return wl_min, wl_max, flx


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
