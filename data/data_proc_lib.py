from scipy import interpolate
from astropy.io import fits

def spec_cln(dir,fnames):
    '''Returns the number of fit files with SPEC_CLN = 2.'''
    gal_count = 0
    spec_cln_names = []
    for fname in fnames:
        with fits.open(dir+fname) as hdul:
            if hdul[0].header['SPEC_CLN'] == 2:
                gal_count += 1
                spec_cln_names.append(fname)

    return gal_count,spec_cln_names

def get_id(fname):
    '''Obtain the fiber id from a file name. Created to use map'''
    id = fname.split('-')[-1][0:3]

    return int(id)


def f_interpolate(x, y, interval):
    # axis = 0 since this is the one containing the slices of the cube
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    return f(interval)

def p_f_interpolate(x, y):
    # axis = 0 since this is the one containing the slices of the cube
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    return f
