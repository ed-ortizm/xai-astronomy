from scipy import interpolate

def f_interpolate(x, y, interval):
    # axis = 0 since this is the one containing the slices of the cube
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    return f(interval)

def p_f_interpolate(x, y):
    # axis = 0 since this is the one containing the slices of the cube
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    return f
