import os
import urllib

from scipy import interpolate
import astropy.io.fits as pyfits
import multiprocessing as mp
import pandas as pd

def spec_cln(dir,fnames):
    '''Returns the number of fit files with SPEC_CLN = 2.'''
    gal_count = 0
    spec_cln_names = []
    for fname in fnames:
        with pyfits.open(dir+fname) as hdul:
            if hdul[0].header['SPEC_CLN'] == 2:
                gal_count += 1
                spec_cln_names.append(fname)

    return gal_count, spec_cln_names

def get_id(fname):
    '''Obtain the fiber id from a file name. Created to use map'''
    id = fname.split('-')[-1][0:3]

    return int(id)


def inpl(x, y, interval):
    '''Interpolation function for the fluxes'''
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    return f(interval)

## From Dovi's student

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
