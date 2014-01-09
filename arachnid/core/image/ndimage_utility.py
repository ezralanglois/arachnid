''' Image enhancement functions

This module contains a set of image enhancement functions using NumPy, SciPy and 
custom wrapped fortran functions (taken from SPIDER).

.. todo:: automatic inversion detection? for defocus? based on ctf

.. Created on Jul 31, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
from eman2_utility import em2numpy2em as _em2numpy2em, em2numpy2res as _em2numpy2res
#import eman2_utility
import analysis
import numpy, scipy, scipy.ndimage
import numpy.fft
import scipy.fftpack, scipy.signal
import scipy.linalg
import scipy.ndimage.filters
import scipy.ndimage.morphology
import scipy.sparse
import scipy.special
from filters import linear
import ndimage_filter
import logging
import math

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from util import _image_utility
    _image_utility;
except:
    tracing.log_import_error('Failed to load _image_utility.so module - certain functions will not be available', _logger)
    _image_utility=None

def mirror(img, out=None):
    ''' Mirror projection (SPIDER convention)
    
    :Parameters:
    
    img : array
          Image
    
    :Returns:
    
    out : array
          Mirrored image
    '''
    
    if img.ndim != 2: raise ValueError, "Mirroring only works with 2D images"
    #xoff = 1 if numpy.mod(img.shape[0], 2) == 0 else 0
    yoff = 1 if numpy.mod(img.shape[1], 2) == 0 else 0
    if out is None: out = img.copy()
    else: out[:]=img[:]
    out[:, yoff:] = numpy.fliplr(img[:, yoff:])
    return out

def mirror_ud(img, out=None):
    ''' Mirror projection (SPIDER convention)
    
    :Parameters:
    
    img : array
          Image
    
    :Returns:
    
    out : array
          Mirrored image
    '''
    
    if img.ndim != 2: raise ValueError, "Mirroring only works with 2D images"
    xoff = 1 if numpy.mod(img.shape[0], 2) == 0 else 0
    #yoff = 1 if numpy.mod(img.shape[1], 2) == 0 else 0
    if out is None: out = img.copy()
    else: out[:]=img[:]
    out[xoff:, :] = numpy.flipud(img[xoff:, :])
    return out

def fourier_space_shift(fimg, dx, dy):
    ''' Shift an image in Fourier space
    
    .. note::
        
        Take from http://www.mathworks.com/matlabcentral/fileexchange/23440-2d-fourier-shift
    
    :Parameters:
    
    img : array
          2D complex array, Fourier transform of image
    dx : float
         Shift in x-direction
    dy : float
         Shift in y-direction
    
    :Returns:
    
    out : array
          2D complex array, shifted Fourier transform of image
    '''
    
    N,M = fimg.shape
    x_shift = numpy.exp(-1j * 2 * numpy.pi * dx * numpy.hstack((numpy.arange(numpy.floor(N/2.0), dtype=numpy.float), numpy.arange(numpy.floor(-N/2.0), 0, dtype=numpy.float))) / N)
    print N, x_shift.shape
    y_shift = numpy.exp(-1j * 2 * numpy.pi * dy * numpy.hstack((numpy.arange(numpy.floor(M/2.0), dtype=numpy.float), numpy.arange(numpy.floor(-M/2.0), 0, dtype=numpy.float))) / M)
    #if numpy.mod(N,2): x_shift[N/2+1] = x_shift[N/2+1].real
    #if numpy.mod(M,2): y_shift[M/2+1] = y_shift[M/2+1].real    
    if numpy.mod(N,2): x_shift[N/2] = x_shift[N/2].real
    if numpy.mod(M,2): y_shift[M/2] = y_shift[M/2].real
    shift = numpy.outer(x_shift, y_shift)
    assert(shift.ndim==fimg.ndim)
    return fimg*shift

def fourier_shift2(img, dx, dy, dz=0, pad=1):
    ''' Shift using sinc interpolation
    
    :Parameters:
    
    img : array
          2D or 3D array of pixels
    dx : float
         Shift in x-direction
    dy : float
         Shift in y-direction
    dz : float
         Shift in z-direction
    pad : float
          Amount of padding
    
    :Returns:
    
    out : array
          2D or 3D array of pixel shift (according to input)
    '''
    
    if img.ndim != 2 and img.ndim != 3: raise ValueError, "Only works with 2 or 3D images"
    if dx == 0 and dy == 0 and dz == 0: return img
    if pad > 1:
        shape = img.shape
        img = pad_image(img.astype(numpy.complex64), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'm')
    if img.ndim == 2:
        fimg = scipy.fftpack.fft2(img)
        fimg = fourier_space_shift(fimg, (dy, dx))
        img = scipy.fftpack.ifftn(fimg).real
    else:
        fimg = scipy.fftpack.fftn(img)
        if img.ndim == 3: fimg = scipy.ndimage.fourier_shift(fimg, (dx, dy, dz))
        else: fimg = scipy.ndimage.fourier_shift(fimg, (dx, dy), -1)
        img = scipy.fftpack.ifftn(fimg).real
    if pad > 1: img = depad_image(img, shape)
    return img

def fourier_shift(img, dx, dy, dz=0, pad=1):
    ''' Shift using sinc interpolation
    
    :Parameters:
    
    img : array
          2D or 3D array of pixels
    dx : float
         Shift in x-direction
    dy : float
         Shift in y-direction
    dz : float
         Shift in z-direction
    pad : float
          Amount of padding
    
    :Returns:
    
    out : array
          2D or 3D array of pixel shift (according to input)
    '''
    
    if img.ndim != 2 and img.ndim != 3: raise ValueError, "Only works with 2 or 3D images"
    if dx == 0 and dy == 0 and dz == 0: return img
    if pad > 1:
        shape = img.shape
        img = pad_image(img.astype(numpy.complex64), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'm')
    if img.ndim == 2:
        fimg = scipy.fftpack.fft2(img)
        fimg = scipy.ndimage.fourier_shift(fimg, (dy, dx), -1, 0)
        img = scipy.fftpack.ifft2(fimg).real
    else:
        fimg = scipy.fftpack.fftn(img)
        if img.ndim == 3: fimg = scipy.ndimage.fourier_shift(fimg, (dx, dy, dz))
        else: fimg = scipy.ndimage.fourier_shift(fimg, (dx, dy), -1)
        img = scipy.fftpack.ifftn(fimg).real
    if pad > 1: img = depad_image(img, shape)
    return img

def integral_image(img):
    '''
    '''
    
    return img.cumsum(1).cumsum(0)

#r/2-1,360
def polar_simple(img, radius=None, angle=360.0):
    ''' Simplest polar transform
    '''
    
    if radius is None: radius = img.shape[0]/2-1
    img = img.astype(numpy.float)
    cx, cy = round(img.shape[0]/2.0), round(img.shape[1]/2.0)
    angles = numpy.arange(0, 2.0*numpy.pi-2.0*numpy.pi/angle, 2.0*numpy.pi/angle)
    out = numpy.zeros((radius, len(angles)))
    for r in xrange(radius):
        for j, a in enumerate(angles):
            rc, rs = cx+round(r*numpy.sin(a)), cy+round(r*numpy.cos(a))
            if rc < out.shape[0] and rs < out.shape[1]:
                out[r, j] = img[rc, rs]
    return out

def radon_transform(nrows, ncols, nangs=180, dtype=numpy.float64):
    '''Compute the radon transform
    
    :Parameters:
    
    nrows : int
            Number of rows in the image
    ncols : int
            Number of columns in the image
    nangs : int
            Number of angles to sample in the image
    dtype : numpy.dtype
            Data type
    
    :Returns:
    
    out : ndarray
          Sinogram image data
    '''
    
    temp1 = nrows - 1 - (nrows-1)/2
    temp2 = ncols - 1 - (ncols-1)/2
    rLast = int(math.ceil(math.sqrt(temp1*temp1+temp2*temp2))) + 1
    rFirst = -rLast
    nrad = rLast - rFirst + 1
    
    n = _image_utility.radon_count(nangs, nrows, ncols)
    try:
        data = numpy.empty(n, dtype=dtype)
        row = numpy.empty(n, dtype=numpy.uint)#numpy.ulonglong)
        col = numpy.empty(n, dtype=numpy.uint)#numpy.ulonglong)
    except:
        logging.error("Error creating sparse matrix with %d"%n)
        raise
    
    n = _image_utility.radon_transform(data, row, col, nangs, nrows, ncols)
    if n < 0: raise StandardError, "Radon transform failed"
    mat = scipy.sparse.coo_matrix( (data[:n],(row[:n], col[:n])), shape=(nrad*nangs, nrows*ncols) )
    mat.nrad = nrad
    mat.nang = nangs
    return mat

def sinogram(img, rad):
    '''Create a sinogram from an image and a radon transform
    
    :Parameters:
    
    img : ndarray
          Image n x m
    rad : ndarray
          Radon transform nm x ra
    
    :Returns:
    
    sino : ndarray
           Sinogram r x a
    '''
    
    sino = rad * img.ravel()[:, numpy.newaxis]
    return sino.reshape(rad.nang, rad.shape[0]/rad.nang)

def frt2(a):
    """Compute the 2-dimensional finite radon transform (FRT) for an n x n
    integer array.
    
    .. note::
        
        http://pydoc.net/scikits-image/0.4.2/skimage.transform.finite_radon_transform
    
    :Parameters:
    
    a : array
        Input image
    
    :Returns:
    
    out : array
          Discreet radon transform
    """
    
    if not issubclass(a.dtype.type, numpy.integer):
        normalize_min_max(a, 0, 2048, a)
        a = a.astype(numpy.int32)
    
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square, 2-D array")
 
    ai = a.copy()
    n = ai.shape[0]
    f = numpy.empty((n+1, n), numpy.uint32)
    f[0] = ai.sum(axis=0)
    for m in xrange(1, n):
        # Roll the pth row of ai left by p places
        for row in xrange(1, n):
            ai[row] = numpy.roll(ai[row], -row)
        f[m] = ai.sum(axis=0)
    f[n] = ai.sum(axis=1)
    return f

def rotavg(img, out=None):
    ''' Create a 2D rotational average of the given image
    
    :Parameters:
    
    img : array
          Image to rotationally average
    out : array
          Output rotationally averaged image
    
    :Returns:
    
    out : array
          Output rotationally averaged image
    '''
    
    img = numpy.asanyarray(img)
    if img.ndim != 2: raise ValueError, "Input array must be 2D"
    if out is None: out = img.copy()
    avg = (img)
    rmax = min(img.shape[0]/2 + img.shape[0]%2, img.shape[1]/2 + img.shape[1]%2)
    if img.ndim > 2: rmax = min(rmax, img.shape[2]/2 + img.shape[2]%2)
    if out.ndim==2: out=out.reshape((out.shape[0], out.shape[1], 1))
    avg = avg.astype(out.dtype)
    assert(rmax <= avg.shape[0])
    _image_utility.rotavg(out, avg, int(rmax))
    if out.shape[2] == 1: out = out.reshape((out.shape[0], out.shape[1]))
    return out

def mean_azimuthal(img, center=None, ret_n=False):
    ''' Calculate the sum of a 2D array along the azimuthal
    
    .. note::
    
        Adopted from https://github.com/numpy/numpy/pull/230/files#r851142
    
    :Parameters:
    
    img : array-like
          Image array
    center : tuple, optional
              Coordaintes of the image center
    
    :Returns:
    
    out : array
          Sum over the radial lines of the image
    '''
    
    img = numpy.asanyarray(img)
    if img.ndim != 2: raise ValueError, "Input array must be 2D: %s"%str(img.shape)
    #img = normalize_min_max(img)*255
    if center is None: center = (img.shape[0]/2+img.shape[0]%2, img.shape[1]/2+img.shape[0]%2)
    i, j = numpy.arange(img.shape[0])[:, None], numpy.arange(img.shape[1])[None, :]   
    i, j = i-center[0], j-center[1]
    k = (j**2+i**2)**.5
    k = k.astype(int)
    if ret_n:
        return numpy.bincount(k.ravel(), img.ravel())/numpy.bincount(k.ravel()), numpy.bincount(k.ravel())
    return numpy.bincount(k.ravel(), img.ravel())/numpy.bincount(k.ravel())

def std_azimuthal(img, center=None):
    ''' Calculate the sum of a 2D array along the azimuthal
    
    .. note::
    
        Adopted from https://github.com/numpy/numpy/pull/230/files#r851142
    
    :Parameters:
    
    img : array-like
          Image array
    center : tuple, optional
              Coordaintes of the image center
    
    :Returns:
    
    out : array
          Sum over the radial lines of the image
    '''
    
    img = numpy.asanyarray(img)
    if img.ndim != 2: raise ValueError, "Input array must be 2D: %s"%str(img.shape)
    
    avg, cnt_n = mean_azimuthal(img, center, ret_n=True)
    
    avg2d = numpy.zeros(img.shape)
    avg2d[len(avg), len(avg):] = avg
    avg2d=rotavg(avg2d)
    img -= avg2d
    
    return (mean_azimuthal(img**2, center) - mean_azimuthal(img, center)**2/cnt_n)/cnt_n

@_em2numpy2res
def find_peaks_fast(cc, width, fwidth=None):
    ''' Find peaks in a cross-correlation map
    
    :Parameters:
    
    cc : array
         Cross-correlation image
    width : float
            Expected width of the peaks
    
    :Returns:
    
    peaks : array (Nx3)
            Array of peaks (peak, x, y)
    '''
    
    if fwidth is None or fwidth < 0: fwidth = width/2.0
    if fwidth > 0.0: cc=scipy.ndimage.filters.gaussian_filter(cc, sigma=fwidth, mode='constant')
    neighborhood = numpy.ones((int(width),int(width)))
    cc_peaks = (scipy.ndimage.filters.maximum_filter(cc, footprint=neighborhood)==cc) - \
                scipy.ndimage.morphology.binary_erosion((cc==0), structure=neighborhood, border_value=1)
    cc_peaks = cc_peaks.ravel()
    offsets, = numpy.nonzero(cc_peaks)
    x,y = numpy.unravel_index(offsets, cc.shape)
    sel = x>=width
    sel = numpy.logical_and(sel, x <= cc.shape[0]-width)
    sel = numpy.logical_and(sel, y >= width)
    sel = numpy.logical_and(sel, y <= cc.shape[1]-width)
    offsets = offsets[sel]
    y,x = numpy.unravel_index(offsets, cc.shape)
    cc = cc.ravel()[offsets].copy().squeeze()
    return numpy.hstack((cc[:, numpy.newaxis], x[:, numpy.newaxis], y[:, numpy.newaxis]))

def grid_image(shape, center=None):
    '''
    '''
    
    if not hasattr(shape, '__iter__'): shape = (shape, shape)
    if center is None: cx, cy = shape[0]/2, shape[1]/2
    elif not hasattr(center, '__iter__'): cx, cy = center, center
    else: cx, cy = center
    #radius2 = radius+1
    y, x = numpy.ogrid[-cx: shape[0]-cx, -cy: shape[1]-cy]
    return x, y

def radial_image(shape, center=None):
    '''
    '''
    
    x, y = grid_image(shape, center)
    return x**2+y**2

def snr_correction_factor(img, ref):
    ''' Calculate an SNR correction factor between a high signal image and
    a noise image. 
    
    :Parameters:
    
    img : array
          2D array of image data for noisy image
    ref : array
          2D array of image data for reference image
    
    :Returns:
    
    factor : float
             Factor for reference image
    '''
    
    return numpy.inner(ref.ravel(), img.ravel())/numpy.linalg.norm(ref, 2)

#@numpy2em_d
def model_disk(radius, shape, center=None, dtype=numpy.int, order='C'):
    ''' Create a disk of given radius with background zero and foreground 1
    
    :Parameters:
    
    shape : int or sequence of two ints
            Shape of the new array, e.g., (2, 2) or 2
    center : int or sequence of two ints, optional
             Center of the disk, if not specified then use the center of the image
    dtype : data-type, optional
            The desired data-type for the array. Default is numpy.int
    order : {'C', 'F'}, optional
            Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) order in memory
    
    :Returns:
    
    img : numpy.ndarray
          Disk image
    
    .. todo:: create numpy2em_d decorator
    
    '''
    
    if not hasattr(shape, '__iter__'): shape = (shape, shape)
    a = numpy.zeros(shape, dtype, order)
    irad = radial_image(shape, center)
    a[irad <= radius**2]=1
    return a

def model_ring(rmin, rmax, shape, center=None, dtype=numpy.int, order='C'):
    ''' Create a disk of given radius with background zero and foreground 1
    
    :Parameters:
    
    shape : int or sequence of two ints
            Shape of the new array, e.g., (2, 2) or 2
    center : int or sequence of two ints, optional
             Center of the disk, if not specified then use the center of the image
    dtype : data-type, optional
            The desired data-type for the array. Default is numpy.int
    order : {'C', 'F'}, optional
            Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) order in memory
    
    :Returns:
    
    img : numpy.ndarray
          Disk image
    
    .. todo:: create numpy2em_d decorator
    
    '''
    
    if not hasattr(shape, '__iter__'): shape = (shape, shape)
    a = numpy.zeros(shape, dtype, order)
    irad = radial_image(shape, center)
    rmin = rmin**2
    rmax = rmax**2
    a[numpy.logical_and(irad > rmin, irad < rmax)]=1
    return a

def acf(img, out=None):
    ''' Autocorrelate an image with itself
    
    :Parameters:
    
    img : array
          Large image to match
    out : array
          Cross-correlation map (same dim as large image)
    
    .. todo:: 
          
        def fft_correlate(A,B,*args,**kwargs):
        return S.signal.fftconvolve(A,B[::-1,::-1,...],*args,**kwargs)
    
    .. todo:: optimize fft
    
    :Returns:
    
    out : array
         Cross-correlation map (same dim as large image)
    '''
    
    if out is None: out = numpy.empty_like(img)
    fp1 = scipy.fftpack.fft2(img)
    numpy.multiply(fp1, fp1.conj(), fp1)
    out[:,:] = numpy.absolute(scipy.fftpack.ifft2(fp1)).real
    return scipy.fftpack.fftshift(out)

def cross_correlate_raw(img, template, phase=False, out=None):
    ''' Cross-correlate an image with a template
    
    :Parameters:
    
    img : array
          Large image to match
    template : array
               Small template to search with
    out : array
          Cross-correlation map (same dim as large image)
    
    :Returns:
    
    cc : array
         Cross-correlation map (same dim as large image)
    '''
    
    template=template.astype(img.dtype)
    out = pad_image(template, img.shape, out=out)
    fp1 = scipy.fftpack.fft2(img)
    fp2 = scipy.fftpack.fft2(out)
    numpy.multiply(fp1, fp2.conj(), fp1)
    if phase:
        fp1 /= numpy.abs(fp1)
    out[:,:] = scipy.fftpack.ifft2(fp1).real
    return out

def cross_correlate(img, template, phase=False, out=None):
    ''' Cross-correlate an image with a template
    
    :Parameters:
    
    img : array
          Large image to match
    template : array
               Small template to search with
    out : array
          Cross-correlation map (same dim as large image)
    
    :Returns:
    
    cc : array
         Cross-correlation map (same dim as large image)
    '''
    
    return scipy.fftpack.fftshift(cross_correlate_raw(img, template, phase, out))

def local_variance(img, mask, out=None):
    ''' Esimtate the local variance on the image, under the given mask
    
    :Parameters:
    
    img : array
          Large image to match
    mask : array
           Small mask under which to estimate variance 
    out : array
          Local variance map (same dim as large image)
    
    :Returns:
    
    cc : array
         Local variance map (same dim as large image)    
    '''
    
    tot = numpy.sum(mask>0)
    mask=normalize_standard(mask, mask, True)*(mask>0)
    mask = pad_image(mask, img.shape)
    shape = img.shape
    img2 = numpy.square(img)
    img2[:,:] = cross_correlate_raw(img2, mask)
    '''numpy.divide(img2, tot, img2)'''
    mask[:,:] = cross_correlate_raw(img, mask)
    del img
    numpy.divide(mask, tot, mask)
    numpy.square(mask, mask)
    numpy.subtract(img2, mask, img2)
    del mask
    img2[img2<=0]=9e20
    numpy.sqrt(img2, img2)
    img2[:,:] = scipy.fftpack.fftshift(img2)
    return depad_image(img2, shape, out)

def rolling_window(array, window=(0,), asteps=None, wsteps=None, intersperse=False):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood defined by window. New dimensions are added at the end of
    `array`, and as no padding is done the arrays first dimensions are smaller
    then before. It is possible to extend only earlier dimensions by giving
    window a 0 sized dimension.
    
    .. note::
    
        Taken from: https://github.com/numpy/numpy/pull/31

    :Parameters:
    
    array : array_like
            Array to which the rolling window is applied.
    window : int or tuple
             Either a single integer to create a window of only the last axis or a
             tuple to create it for the last len(window) axes. 0 can be used as a
             to ignore a dimension in the window.
    asteps : tuple
             Aligned at the last axis, new steps for the original array, ie. for
             creation of non-overlapping windows.
    wsteps : int or tuple (same size as window)
             steps inside the window this can be 0 to repeat along the axis.
    intersperse : bool
                  If True, the new dimensions are right after the corresponding original
                  dimension, instead of at the end of the array.

    :Returns:
    
    out : array
          A view on `array` which is smaller to fit the windows and has windows added
          dimensions (0s not counting), ie. every point of `array` is an array of size
          window.
    
    Examples:
    
        >>> a = numpy.arange(16).reshape(4,4)
        >>> rolling_window(a, (2,2))[0:2,0:2]
        array([[[[ 0,  1],
                 [ 4,  5]],
    
                [[ 1,  2],
                 [ 5,  6]]],
    
    
               [[[ 4,  5],
                 [ 8,  9]],
    
                [[ 5,  6],
                 [ 9, 10]]]])

    Or to create non-overlapping windows, but only along the first dimension:
        >>> rolling_window(a, (2,0), asteps=(2,1))
        array([[[ 0,  4],
                [ 1,  5],
                [ 2,  6],
                [ 3,  7]],
    
               [[ 8, 12],
                [ 9, 13],
                [10, 14],
                [11, 15]]])
    Note that the 0 is discared, so that the output dimension is 3:
        >>> rolling_window(a, (2,0), asteps=(2,1)).shape
        (2, 4, 2)

    This is useful for example to calculate the maximum in all 2x2 submatrixes:
        >>> rolling_window(a, (2,2), asteps=(2,2)).max((2,3))
        array([[ 5,  7],
               [13, 15]])

    Or delay embedding (3D embedding with delay 2):
        >>> x = numpy.arange(10)
        >>> rolling_window(x, 3, wsteps=2)
        array([[0, 2, 4],
               [1, 3, 5],
               [2, 4, 6],
               [3, 5, 7],
               [4, 6, 8],
               [5, 7, 9]])
    """
    array = numpy.asarray(array)
    orig_shape = numpy.asarray(array.shape)
    window = numpy.atleast_1d(window).astype(int) # maybe crude to cast to int...

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if numpy.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension. - %d < %d"%(len(array.shape), len(window)))

    _asteps = numpy.ones_like(orig_shape)
    if asteps is not None:
        asteps = numpy.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if numpy.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = numpy.ones_like(window)
    if wsteps is not None:
        wsteps = numpy.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if numpy.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if numpy.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension. - %s -- %d*%d=%d"%(str(orig_shape[-len(window):]), window, wsteps, window*wsteps))

    new_shape = orig_shape # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = numpy.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if not intersperse:
        new_shape = numpy.concatenate((shape, window))
        new_strides = numpy.concatenate((strides, new_strides))
    else:
        _ = numpy.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = numpy.zeros(len(shape)*2, dtype=int)
        new_strides = numpy.zeros(len(shape)*2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return numpy.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

def powerspec1d(img):
    ''' Calculated a 1D power spectra of an image
    
    :Parameters:
    
    img : array
          Image
    
    :Returns:
    
    roo : array
          1D rotational average of the power spectra
    '''
    
    fimg = numpy.fft.fftn(img)
    fimg = fimg*fimg.conjugate()
    return mean_azimuthal(numpy.abs(numpy.fft.fftshift(fimg)))[1:fimg.shape[0]/2]

def multitaper_power_spectra(mic, half_nbw=9, low_bias=False, shift=True):
    '''
    '''
    
    import _multitaper
    n = min(mic.shape)
    mic_sq = mic[:n, :n]
    n_tapers_max = int(2 * half_nbw)
    dpss2, eigvals = _multitaper.dpss_windows(mic_sq.shape[1], half_nbw, n_tapers_max,low_bias=low_bias)
    _logger.info("Number of basis functions for %f bandwidth and %d tapers: %d"%(half_nbw, n_tapers_max, dpss2.shape[0]))
    pow = mic_sq.copy()
    pow[:]=0
    mic_sq = mic_sq - numpy.mean(mic_sq)#, axis=-1)[:, numpy.newaxis]
    weights = numpy.sqrt(eigvals)
    for i in xrange(dpss2.shape[0]):
        for j in xrange(dpss2.shape[0]):
            _logger.error("i: %d < %d - j: %d < %d"%(i, dpss2.shape[0], j, dpss2.shape[0]))
            tmp = numpy.outer(dpss2[i], dpss2[j])
            fmic = scipy.fftpack.fft2(mic_sq*tmp)
            pow += numpy.abs(weights[i]*weights[j]*fmic)**2
    pow *= 2 / numpy.sum(numpy.abs(weights[:, numpy.newaxis,numpy.newaxis]) ** 2, axis=-3)
    return numpy.fft.fftshift(pow).copy() if shift else pow.copy()

def perdiogram(mic, window_size=256, pad=1, overlap=0.5, offset=0.1, shift=True, ret_more=False):
    '''
    '''
    
    if offset > 0 and offset < 1.0: offset = int(offset*mic.shape[0])
    step = max(1, window_size*overlap)
    rwin = rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
    rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
    return powerspec_avg(rwin, pad, shift) if not ret_more else (powerspec_avg(rwin, pad, shift), rwin.shape[0])

def dct_avg(imgs, pad):
    ''' Calculate an averaged power specra from a set of images
    
    :Parameters:
    
    imgs : iterable
           Iterator of images
    pad : int
          Number of times to pad an image
    
    :Returns:
    
    avg_powspec : array
                  Averaged power spectra
    '''
    
    if pad is None or pad <= 0: pad = 1
    avg = None
    total = 0.0
    for img in imgs:
        #pad_width = img.shape[0]*pad
        
        fimg=scipy.fftpack.dct(scipy.fftpack.dct(img.T, axis=-1, norm='ortho').T, axis=-2, norm='ortho')
        
        if avg is None: avg = fimg.copy()
        else: avg += fimg
        return avg
        total += 1.0
    numpy.divide(avg, total, avg)
    return avg #scipy.fftpack.fftshift(avg).real

def powerspec_sum(imgs, pad, avg=None, total=0.0, do_ramp=False):
    ''' Calculate an averaged power specra from a set of images
    
    :Parameters:
    
    imgs : iterable
           Iterator of images
    pad : int
          Number of times to pad an image
    
    :Returns:
    
    avg_powspec : array
                  Averaged power spectra
    '''
    if pad is None or pad <= 0: pad = 1
    for img in imgs:
        pad_width = img.shape[0]*pad
        #if eman2_utility.EMAN2 is not None:
        img = img.copy()
        if do_ramp:
            ndimage_filter.ramp(img, img)
        img -= img.min()
        img /= img.max()
        img -= img.mean()
        img /= img.std()
        if img.ndim == 2:
            fimg = numpy.fft.fft2(pad_image(img, (pad_width, pad_width), 'e'))
        else:
            fimg = numpy.fft.fftn(pad_image(img, (pad_width, pad_width), 'e'))
        fimg = fimg*fimg.conjugate()
        if avg is None: avg = fimg.copy()
        else: avg += fimg
        total += 1.0
    return avg, total

def powerspec_fin(avg, total, shift=True):
    '''
    '''
    
    avg = numpy.abs(numpy.fft.fftshift(avg).copy()) if shift else numpy.abs(avg)
    numpy.sqrt(avg, avg)
    numpy.divide(avg, total, avg)
    return avg

def powerspec_avg(imgs, pad, shift=True):
    ''' Calculate an averaged power specra from a set of images
    
    :Parameters:
    
    imgs : iterable
           Iterator of images
    pad : int
          Number of times to pad an image
    
    :Returns:
    
    avg_powspec : array
                  Averaged power spectra
    '''
    
    avg, total = powerspec_sum(imgs, pad)
    return powerspec_fin(avg, total, shift)

def moving_average(img, window=3, out=None):
    ''' Estimate a moving average with a uniform distribution and given window size
    
    :Parameters:
    
    img : array
          1-D rotational average
    window : int
             Window size
    out : array, optional
          Output array
    
    :Returns:
    
    out : array, optional
          Output array
    '''
    
    if out is None: out = img.copy()
    off = int(window/2.0)
    if 1 == 1:
        weightings = numpy.ones(window)
        weightings /= weightings.sum()
        out[off:len(img)-off]=numpy.convolve(img, weightings)[window-1:-(window-1)]
        return out
    b = rolling_window(img, window)
    avg = b.mean(axis=-1)
    out[off:len(img)-off] = avg
    out[:off] = avg[0]
    out[off:] = avg[len(avg)-1]
    return out

def moving_minimum(img, window=3, out=None):
    ''' Estimate a moving average with a uniform distribution and given window size
    
    :Parameters:
    
    img : array
          1-D rotational average
    window : int
             Window size
    out : array, optional
          Output array
    
    :Returns:
    
    out : array, optional
          Output array
    '''
    
    if out is None: out = img.copy()
    off = int(window/2.0)
    b = rolling_window(img, window)
    avg = b.mean(axis=1)
    out[off:len(img)-off] = avg
    out[:off] = avg[0]
    out[off:] = avg[len(avg)-1]
    return out

def local_variance2(img, mask, out=None):
    ''' Esimtate the local variance on the image, under the given mask
    
    .. todo:: Fix this!
    
    :Parameters:
    
    img : array
          Large image to match
    mask : array
           Small mask under which to estimate variance 
    out : array
          Local variance map (same dim as large image)
    
    :Returns:
    
    cc : array
         Local variance map (same dim as large image)    
    '''
    
    #normalize mask
    #then mask mask
    
    tot = numpy.sum(mask>0) #img.ravel().shape[0]
    mask=normalize_standard(mask, mask, True)*(mask>0)
    shape = (img.shape[0]+mask.shape[0], img.shape[1]+mask.shape[1])
    img2 = pad_image(img, shape)
    mask = pad_image(mask, shape)
    shape = img.shape
    img = img2.copy()
    numpy.square(img, img2)
    img2[:,:] = cross_correlate_raw(img2, mask)
    '''numpy.divide(img2, tot, img2)'''
    mask[:,:] = cross_correlate_raw(img, mask)
    del img
    numpy.divide(mask, tot, mask)
    numpy.square(mask, mask)
    numpy.subtract(img2, mask, img2)
    del mask
    img2[img2<=0]=9e20
    numpy.sqrt(img2, img2)
    img2[:,:] = scipy.fftpack.fftshift(img2)
    return depad_image(img2, shape, out)

def depad_image(img, shape, out=None):
    ''' Depad an image
    
    :Parameters:
    
    img : array
          Image to pad
    shape : tuple
            Dimensions of new image
    out : array
          Padded image
    
    :Returns:
    
    out : array
          Padded image
    '''
    
    if out is None: 
        if img.shape[0] == shape[0] and img.shape[1] == shape[1]: return img
        out = numpy.zeros(shape)
    cx = (img.shape[0]-shape[0])/2
    cy = (img.shape[1]-shape[1])/2
    out[:,:] = img[cx:cx+shape[0], cy:cy+shape[1]]
    return out

def pad_image(img, shape, fill=0.0, out=None):
    ''' Pad an image with zeros
    
    :Parameters:
    
    img : array
          Image to pad
    shape : tuple
            Dimensions of new image
    fill : float
           Value to fill array with
    out : array
          Padded image
    
    :Returns:
    
    out : array
          Padded image
    '''
    
    cx = (shape[0]-img.shape[0])/2
    cy = (shape[1]-img.shape[1])/2
    if out is None: 
        if img.shape[0] == shape[0] and img.shape[1] == shape[1]: return img
        out = numpy.zeros(shape, dtype=img.dtype)
        if fill == 'm':
            out[0:cx, cy:cy+img.shape[1]] = img[-cx:0:-1, :]
            out[cx+img.shape[0]:, cy:cy+img.shape[1]] = img[-cx:0:-1, :]
            out[cx:cx+img.shape[0], 0:cy] = img[:, 0:cx]
            out[cx:cx+img.shape[0], cy+img.shape[1]:] = img[:, 0:cx]
        elif fill == 'e': 
            out[:, :] = (img[0, :].sum()+img[:, 0].sum()+img[len(img)-1, :].sum()+img[:, len(img)-1].sum()) / (img.shape[0]*2+img.shape[1]*2 - 4)
        elif fill == 'r': out[:, :] = numpy.random.normal(img.mean(), img.std(), shape)
        elif fill != 0: out[:, :] = fill
    out[cx:cx+img.shape[0], cy:cy+img.shape[1]] = img
    return out

def filter_annular_bp(img, freq1, freq2):
    ''' Filter an image with a Gaussian low pass filter
    
    Todo: optimize kernel
    
    :Parameters:
    
    img : array
          Image to filter
    sigma : float
            Cutoff frequency
    out : array
          Filtered image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    img = img.astype(numpy.complex64) # todo percison based on input
    kernel = numpy.zeros(img.shape, dtype=img.dtype)
    irad = radial_image(img.shape)
    val =  (1.0/(1.0+numpy.exp(((numpy.sqrt(irad)-freq1))/(10.0))))*(1.0-(numpy.exp(-irad /(2*freq2*freq2))))
    kernel[:, :].real = val
    kernel[:, :].imag = val
    return filter_image(img, kernel)


def spiral_transform(img):
    '''
    Todo: optimize kernel
    '''
    
    cx, cy = img.shape[0]/2, img.shape[1]/2
    img = img.astype(numpy.complex64) # todo percison based on input
    kernel = numpy.zeros(img.shape, dtype=img.dtype)
    for i in xrange(kernel.shape[0]):
        for j in xrange(kernel.shape[1]):
            v1, v2 = i-cx, j-cy
            if v1 == 0 and v2 == 0: continue
            kernel[i,j] = numpy.complex(v2,v1)/numpy.sqrt( numpy.power(float(v1), 2)+numpy.power(float(v2), 2) )
    return filter_image(img, kernel)
    

def filter_gaussian_lp(img, sigma, out=None):
    ''' Filter an image with a Gaussian low pass filter
    
    :Parameters:
    
    img : array
          Image to filter
    sigma : float
            Cutoff frequency
    out : array
          Filtered image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    sigma = 0.5 / sigma / sigma
    if hasattr(scipy.ndimage.filters, 'fourier_gaussian'):
        return scipy.ndimage.filters.fourier_gaussian(img, sigma, output=out)
    return scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')#, output=out)

def filter_butterworth_lowpass(img, low_cutoff, falloff, pad=1):
    ''' Apply a Butterworth lowpass filter to an image
    
    .. seealso:: :py:func:`linear.butterworth_lowpass`
    
    :Parameters:
    
    img : array
          Image
    low_cutoff : float
                 Low-frequency cutoff
    falloff : float
              Low-frequency fall off
    pad : int
          Padding
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad > 1:
        shape = img.shape
        ctype = numpy.complex128 if img.dtype is numpy.float64 else numpy.complex64
        img = pad_image(img.astype(ctype), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'e')
    img = filter_image(img, linear.butterworth_lowpass(img.shape, low_cutoff, falloff), pad)
    if pad > 1: img = depad_image(img, shape)
    return img

def filter_image(img, kernel, pad=1):
    '''
    .. todo:: filter padding
    '''
        
    #fimg = scipy.fftpack.fftn(img)
    #kernel = scipy.fftpack.ifftshift(kernel)
    #numpy.multiply(fimg, kernel, fimg)
    #return scipy.fftpack.ifftn(fimg).real

    fimg = scipy.fftpack.fftshift(scipy.fftpack.fftn(img))
    numpy.multiply(fimg, kernel, fimg)
    return scipy.fftpack.ifftn(scipy.fftpack.ifftshift(fimg)).real.copy()

@_em2numpy2em
def compress_image(img, mask, out=None):
    ''' Compress the valid region of an image with the given mask into 1D array
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Binary mask of valid pixesl
    out : numpy.ndarray
          Output 1D-array
    
    :Returns:
        
    out : numpy.ndarray
          Output 1D-array
    '''
    
    if out is None:
        out = img.ravel()[mask.ravel()>0.5].copy()
    else:
        out[:] = img.ravel()[mask.ravel()>0.5]
    return out

def uncompress_image(img, mask, out=None):
    ''' Compress the valid region of an image with the given mask into 1D array
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Binary mask of valid pixesl
    out : numpy.ndarray
          Output 1D-array
    
    :Returns:
        
    out : numpy.ndarray
          Output 1D-array
    '''
    
    if out is None:
        out = numpy.zeros(mask.shape, img.dtype)
    out.ravel()[mask.ravel()>0.5] = img
    return out

@_em2numpy2em
def fftamp(img, s=None, out=None):
    ''' Calculate the power spectra of an image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
        
    :Returns:
    
    out : numpy.ndarray
          Normalized image
    '''
    
    
    #fimg = scipy.fftpack.fft2(img, s)
    #fimg = scipy.fftpack.fftshift(fimg)
    fimg = numpy.fft.fftn(img, s)
    fimg = numpy.fft.fftshift(fimg)
    fimg = fimg*fimg.conjugate()
    out = numpy.abs(fimg, out)
    return out

def polar(image, center=None, out=None, rng=None):
    '''Transform image into log-polar representation
    
    @todo - add radius range
    
    :Parameters:
    
    image : numpy.ndarray
            Input image
    angles : int, optional
             Size of angle dimension
    out : numpy.ndarray, optional
          Image in log polar space
    
    :Returns:
    
    out : numpy.ndarray
          Image in polar space (radius, angle)
    '''
    
    ny, nx = image.shape[:2]
    if center is None: center = (nx // 2, ny // 2)
    x, y = index_coords(image, center)
    r, theta = cart2polar(x, y)
    
    if rng is None: rng = (r.min(), r.max())
    r_i = numpy.linspace(rng[0], rng[1], nx)
    theta_i = numpy.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = numpy.meshgrid(theta_i, r_i)
    
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += center[0] # We need to shift the origin back to 
    yi += center[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = numpy.vstack((xi, yi))

    return scipy.ndimage.interpolation.map_coordinates(image, coords).reshape((nx, ny)).T
    #return out

def polar_to_cart(image, center=None, out=None, rng=None, order=0):
    '''Transform image into log-polar representation
    
    Does not work!
    
    :Parameters:
    
    image : numpy.ndarray
            Input image
    angles : int, optional
             Size of angle dimension
    out : numpy.ndarray, optional
          Image in log polar space
    
    :Returns:
    
    out : numpy.ndarray
          Image in polar space (radius, angle)
    '''
    
    ny, nx = image.shape[:2]
    if center is None: center = (nx // 2, ny // 2)
    x, y = index_coords(image, center)
    r, theta = cart2polar(x, y)
    
    if rng is None: rng = (r.min(), r.max())
    r_i = numpy.linspace(rng[0], rng[1], nx)
    theta_i = numpy.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = numpy.meshgrid(theta_i, r_i)
    xi, yi = polar2cart(r_grid, theta_grid)
    r, theta = cart2polar(xi, yi)
    xi, yi = r.flatten(), theta.flatten()
    coords = numpy.vstack((xi, yi))

    return scipy.ndimage.interpolation.map_coordinates(image, coords).reshape((nx, ny)).T

def cart2polar(x, y):
    r = numpy.sqrt(x**2 + y**2)
    theta = numpy.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * numpy.cos(theta)
    y = r * numpy.sin(theta)
    return x, y

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    x, y = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))
    x -= origin[0]
    y -= origin[1]
    return x, y

def logpolar(image, angles=None, radii=None, center=None, out=None):
    '''Transform image into log-polar representation
    
    :Parameters:
    
    image : numpy.ndarray
            Input image
    angles : int, optional
             Size of angle dimension
    radii : int, optional
             Size of radius dimension
    out : numpy.ndarray, optional
          Image in log polar space
    
    :Returns:
    
    out : numpy.ndarray
          Image in log polar space
    log_base : float
               Log base for image
    '''
    
    shape = image.shape
    if center is None: center = shape[0] / 2, shape[1] / 2
    if angles is None: angles = shape[0]
    if radii is None:  radii = shape[1]
    theta = numpy.empty((angles, radii), dtype=numpy.float64)
    theta.T[:] = -numpy.linspace(0, numpy.pi, angles, endpoint=False)
    #d = radii
    d = numpy.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = numpy.empty_like(theta)
    radius[:] = numpy.power(log_base, numpy.arange(radii, dtype=numpy.float64)) - 1.0
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    if out is None: out = numpy.empty_like(x)
    scipy.ndimage.interpolation.map_coordinates(image, [x, y], output=out)
    return out, log_base

@_em2numpy2em
def fourier_mellin(img, out=None):
    ''' Calculate the Fourier Mellin transform of an image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
        
    :Returns:
    
    out : numpy.ndarray
          Normalized image
    '''
    
    fimg = fftamp(img)
    fimg = logpolar(fimg)[0]
    fimg = numpy.fft.fft2(fimg)
    out = numpy.abs(numpy.fft.fftshift(fimg), out)
    return out

@_em2numpy2em
def segment(img, bins=0, mask=None, out=None):
    ''' Segment the given image with Otsu's method
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    bins : int
           Number of bins to use in discretizing data for Otsu's method
    mask : array, optional
           Mask to estimate threshold
    out : numpy.ndarray
          Output image
        
    :Returns:
    
    out : numpy.ndarray
          Normalized image
    '''
    
    if mask is not None:
        th = analysis.otsu(compress_image(img, mask).ravel(), bins)
    else:
        th = analysis.otsu(img.ravel(), bins)
    return numpy.greater(img, th, out)

@_em2numpy2em
def dog(img, pixel_radius, dog_width=1.2, out=None):
    ''' Calculate difference of Gaussian over the given image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    pixel_radius : int
                   Radius of the particle in pixels
    dog_width : float
                Width of the difference of Gaussian
    out : numpy.ndarray
          Output image
        
    :Returns:
    
    out : numpy.ndarray
          Normalized image
    
    ..note::
        
        The parameter estimation comes from:
        
            `DoG Picker and TiltPicker: software tools to facilitate particle selection in single particle electron microscopy`
            doi:  10.1016/j.jsb.2009.01.004
            http://ami.scripps.edu/redmine/projects/ami/wiki/DoGpicker
    '''
    
    kfact = math.sqrt( (dog_width**2 - 1.0) / (2.0 * dog_width**2 * math.log(dog_width)) )
    sigma1 = (kfact * pixel_radius)
    sigmaDiff = sigma1*math.sqrt(dog_width*dog_width-1.0)
    #sigma1 = math.sqrt(sigma1**2 + sigmaDiff**2)
    dlst = scipy.ndimage.gaussian_filter(img, sigma=sigma1)
    dnxt = scipy.ndimage.gaussian_filter(dlst, sigma=sigmaDiff)# C3100-08393194-001
    return numpy.subtract(dlst, dnxt, out)

@_em2numpy2em
def normalize_standard(img, mask=None, var_one=True, out=None):
    ''' Normalize image to zero mean and one variance
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Mask for mean/std calculation
    var_one : bool
              Set False to disable variance one normalization
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    mdata = img[mask>0.5] if mask is not None else img
    std = numpy.std(mdata) if var_one else 0.0
    out = numpy.subtract(img, numpy.mean(mdata), out)
    if std != 0.0: numpy.divide(out, std, out)
    return out

@_em2numpy2em
def normalize_standard_norm(img, mask=None, var_one=True, dust_sigma=2.5, xray_sigma=None, replace=None, out=None):
    ''' Normalize image to zero mean and one variance
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Mask for mean/std calculation
    var_one : bool
              Set False to disable variance one normalization
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    mdata = img[mask>0.5] if mask is not None else img
    mdata=replace_outlier(mdata, dust_sigma, xray_sigma, replace, mdata)
    out = numpy.subtract(img, numpy.mean(mdata), out)
    if var_one:
        std = numpy.std(mdata)
        if std != 0.0: numpy.divide(out, std, out)
    return out

@_em2numpy2em
def normalize_min_max(img, lower=0.0, upper=1.0, mask=None, out=None):
    ''' Normalize image to given lower and upper range
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    lower : float
            Lower value
    upper : numpy.ndarray
            Upper value
    mask : numpy.ndarray
           Mask for min/max calculation
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    if numpy.issubdtype(img.dtype, numpy.integer):
        img = img.astype(numpy.float)
    
    vmin = numpy.min(img) if mask is None else numpy.min(img*mask)
    out = numpy.subtract(img, vmin, out)
    vmax = numpy.max(img) if mask is None else numpy.max(img*mask)
    if vmax == 0: raise ValueError, "No information in image"
    numpy.divide(out, vmax, out)
    upper = upper-lower
    if upper != 1.0: numpy.multiply(upper, out, out)
    if lower != 0.0: numpy.add(lower, out, out)
    return out

@_em2numpy2em
def invert(img, out=None):
    '''Invert an image
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Output image
    '''
        
    out = numpy.multiply(img, -1.0, out)
    normalize_min_max(out, -numpy.max(out), -numpy.min(out), out)
    return out

@_em2numpy2em
def vst(img, out=None):
    ''' Variance stablizing transform
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    mval = numpy.min(img)
    if mval < 0: out = numpy.subtract(img, mval, out)
    out = numpy.add(numpy.sqrt(img), numpy.sqrt(img+1), out)
    return out

@_em2numpy2em
def replace_outlier(img, dust_sigma, xray_sigma=None, replace=None, out=None):
    '''Clamp outlier pixels, either too black due to dust or too white due to hot-pixels. Replace with 
    samples drawn from the normal distribution with the same mean and standard deviation.
    
    Any random values drawn outside the range of values in the image are clamped to the largest value.
    
    :Parameters:

    img : numpy.ndarray
          Input image
    dust_sigma : float
                 Number of standard deviations for black pixels
    xray_sigma : float
                 Number of standard deviations for white pixels
    replace : float
              Value to replace with, if None use random noise
    out : numpy.ndarray
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if out is None: out = img.copy()
    else: out[:]=img
    avg = numpy.mean(img)
    std = numpy.std(img)
    vmin = numpy.min(img)
    vmax = numpy.max(img)
    if vmin == vmax: 
        #print "*****", vmin, '==', vmax
        return out

    
    if xray_sigma is None: xray_sigma=dust_sigma if dust_sigma > 0 else -dust_sigma
    if dust_sigma > 0: dust_sigma = -dust_sigma
    lcut = avg+std*dust_sigma
    hcut = avg+std*xray_sigma
    try:
        vsmin = numpy.min(out[img>=lcut])
    except: 
        vsmin=numpy.min(out)
    try:
        vsmax = numpy.max(out[img<=hcut])
    except: 
        vsmax=numpy.max(out)
    if replace == 'mean':
        replace = numpy.mean(out[numpy.logical_and(out > lcut, out < hcut)])
    if vmin < lcut:
        sel = img < lcut
        if replace is None:
            try:
                out[sel] = numpy.random.normal(avg, std, numpy.sum(sel)).astype(out.dtype)
            except:
                v = numpy.random.normal(avg, std, numpy.sum(sel))
                _logger.error("%s -- %s -- %s -- %s"%(str(out.shape), str(out.dtype), str(v.shape), str(v.dtype)))
                raise
        else: out[sel] = replace
    if vmax > hcut:
        sel = img > hcut
        if replace is None:
            out[sel] = numpy.random.normal(avg, std, numpy.sum(sel)).astype(out.dtype)
        else: out[sel] = replace
    out[img > vsmax]=vsmax
    out[img < vsmin]=vsmin
    return out

@_em2numpy2em
def crop_window(img, x, y, offset, out=None):
    ''' Extract a square window from an image
    
    .. todo:: requires testing for wrap around!
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    x : int
        Center of window on x-axis
    y : int
        Center of window on y-axis
    offset : int
             Half-width of the window
    out : numpy.ndarray
          Output window
                     
    :Returns:
    
    out : numpy.ndarray
          Output window
    '''
    
    offset=int(offset)
    xb = int(x)-offset
    yb = int(y)-offset
    width = offset*2
    xe = xb+width
    ye = yb+width
    
    if out is None: out = numpy.zeros((width, width), dtype=img.dtype)
    dxb, dyb, dxe, dye = 0, 0, out.shape[1], out.shape[0]
    dx, dy = dxe, dye
    
    if xb < 0:
        dxb = -xb
        xb = 0
    if yb < 0:
        dyb = -yb
        yb = 0
    if img.shape[1] < xe: xe = img.shape[1]
    if img.shape[0] < ye: ye = img.shape[0]
    dxe = dxb+(xe-xb)
    dye = dyb+(ye-yb)

    out[dyb:dye, dxb:dxe] = img[yb:ye, xb:xe]
    if dxb > 0: out[dyb:dye, :dxb] = img[yb:ye, img.shape[1]-dxb:]
    if dyb > 0: out[:dyb, dxb:dxe] = img[img.shape[0]-dyb:, xb:xe]
    if dxe < out.shape[1]: out[dyb:dye, dxe:] = img[yb:ye, :out.shape[1]-dxe]
    if dye < out.shape[0]: out[dye:, dxb:dxe] = img[:out.shape[0]-dye, xb:xe]
    return out

@_em2numpy2em
def crop_window_old(img, x, y, offset, out=None):
    ''' Extract a square window from an image
    
    .. todo:: requires testing for wrap around!
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    x : int
        Center of window on x-axis
    y : int
        Center of window on y-axis
    offset : int
             Half-width of the window
    out : numpy.ndarray
          Output window
                     
    :Returns:
    
    out : numpy.ndarray
          Output window
    '''

    
    xb = x-offset
    yb = y-offset
    width = offset*2
    xe = xb+width
    ye = yb+width
    
    if out is None: out = numpy.zeros((width, width), dtype=img.dtype)
    dxb, dyb, dxe, dye = 0, 0, 0, 0
    if xb < 0:
        dxb = -xb
        xb = 0
    if yb < 0:
        dyb = -yb
        yb = 0
    if img.shape[1] < xe: 
        dxe = xe-img.shape[1]
        xe = img.shape[1]
    if img.shape[0] < ye:
        #img.shape[0]-yb
        dye = ye-img.shape[0]
        ye = img.shape[0]
    
    try:
        out[dyb:width-dye, dxb:width-dxe] = img[yb:ye, xb:xe]
        '''
        xb: 894, xe:1009 - yb: 3994, ye: 4096 | dxb:0, dxe:0 - dyb:0, dye:13
        (102,115) into shape (101,115)
        4096x4096
        
        '''
    except:
        _logger.error("Error in window1 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dxe > 0: out[dyb:width-dye, width-dxe:] = img[yb:ye, xb-dxe:xb]
        #if dxe > 0: out[dyb:width-dye, width-dxe:] = img[yb:ye, xb-dxe-1:xb]
        # (78,6) xb-dxe:xb = 856-7:856
        # into shape 
        # (78,7) width-dxe = 78 - 7:78  71:78
        
        #1773,2091 - 3433,3710 | 0,0 - 0,41 - width: 318
    except:
        _logger.error("Error in window2 xb:%d,%d - yb:%d,%d | dxb:%d,%d - dyb:%d,%d - width: %d -- out: %d -- img: %d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye, width, out.shape[0], img.shape[0]))
        raise
    try:
        # ValueError: could not broadcast input array from shape (3,74) into shape (2,74)
        # 2013-11-22 09:00:00,812 ERROR Error in window3 344,418 - 856,927 | 0,0 - 0,3
        #yb=856
        #ye=927
        #dyb=0
        #dye=3
        #if dye > 0: out[width-dye:, dxb:width-dxe] = img[yb-dye-1:yb, xb:xe]
        if dye > 0: out[width-dye:, dxb:width-dxe] = img[yb-dye:yb, xb:xe]
    except:
        _logger.error("Error in window3 %d,%d - %d,%d | %d,%d - %d,%d - width: %d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye, width))
        raise
    try:
        if dyb > 0: out[:dyb, dxb:width-dxe] = img[ye:ye+dyb, xb:xe]
    except:
        _logger.error("Error in window4 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dxb > 0: out[dyb:width-dye, :dxb] = img[yb:ye, xe:xe+dxb]
    except:
        _logger.error("Error in window5 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    return out

@_em2numpy2em
def crop_window2(img, x, y, offset, out=None):
    ''' Extract a square window from an image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    x : int
        Center of window on x-axis
    y : int
        Center of window on y-axis
    offset : int
             Half-width of the window
    out : numpy.ndarray
          Output window
                     
    :Returns:
    
    out : numpy.ndarray
          Output window
    '''
    
    if not hasattr(offset, '__iter__'):
        offset = (offset, offset)
    
    xb = x-offset[0]/2
    yb = y-offset[1]/2
    xe = xb+offset[0]
    ye = yb+offset[1]
    
    if out is None: out = numpy.zeros(offset, dtype=img.dtype)
    dxb, dyb, dxe, dye = 0, 0, 0, 0
    if img.shape[1] < xe: 
        dxe = xe-img.shape[1]
        xe = img.shape[1]
    if img.shape[0] < ye: 
        dye = ye-img.shape[0]
        ye = img.shape[0]
    if xb < 0:
        dxb = -xb
        xb = 0
    if yb < 0:
        dyb = -yb
        yb = 0
    
    try:
        out[dyb:offset[0]-dye, dxb:offset[1]-dxe] = img[yb:ye, xb:xe]
    except:
        _logger.error("Error in window1 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dxe > 0: out[dyb:offset[0]-dye, offset[1]-dxe:] = img[yb:ye, xb-dxe:xb]
    except:
        _logger.error("Error in window2 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dye > 0: out[offset[0]-dye:, dxb:offset[1]-dxe] = img[yb-dye:yb, xb:xe]
    except:
        _logger.error("Error in window3 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dyb > 0: out[:dyb, dxb:offset[1]-dxe] = img[ye:ye+dyb, xb:xe]
    except:
        _logger.error("Error in window4 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dxb > 0: out[dyb:offset[0]-dye, :dxb] = img[yb:ye, xe:xe+dxb]
    except:
        _logger.error("Error in window5 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    return out

@_em2numpy2em
def for_each_window(mic, coords, window, bin_factor=1.0):
    ''' Extract a window from a micrograph for each coordinate in this list
    
    :Parameters:
        
    mic : numpy.ndarray
          Micrograph image
    coords : list
             List of coordinates to center of particle
    window : int
             Size of the window to be cropped
    bin_factor : float
                 Number of times to downsample the coordinates
    
    :Returns:
    
    win : numpy.ndarray
          Window from the micrograph
    '''
    
    npdata = numpy.zeros((window, window))
    offset = window/2
    for coord in coords:
        (x, y) = (coord.x, coord.y) if hasattr(coord, 'x') else (coord[1], coord[2])
        crop_window(mic, int(float(x)/bin_factor), int(float(y)/bin_factor), offset, npdata)
        yield npdata
    raise StopIteration

def flatten_solvent(img, threshold=None, out=None):
    ''' Flatten the solven around the structure
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    threshold : float, optional
                Threshold for binarization, if not specified us Otsu's method to find
    out : numpy.ndarray, optional
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    threshold : float
                Threshold used to create binary mask
    '''
    
    if img.ndim != 2 and img.ndim != 3: raise ValueError, "Requires 2 or 3 dimensional images"
    _logger.debug("Tight mask - started")
    if threshold is None or threshold == 'A': 
        _logger.debug("Finding threshold")
        threshold = analysis.otsu(img.ravel())
    else: threshold=float(threshold)
    
    _logger.debug("Finding biggest object")
    # Get largest component in the binary image
    #print "threshold=", threshold
    if 1 == 0:
        out = biggest_object(img>threshold, out)
    else:
        if out is None: out = numpy.zeros(img.shape, img.dtype)
        sel = biggest_object_select(img>threshold)
        out[sel] = img[sel]
    return out, threshold

def tight_mask(img, threshold=None, ndilate=1, gk_size=3, gk_sigma=3.0, out=None):
    ''' Create a tight mask from the given image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    threshold : float, optional
                Threshold for binarization, if not specified us Otsu's method to find
    ndilate : int, optional
              Number of times to dilate the binary image
    gk_size : int, optional
              Size of Gaussian kernel used for real space smoothing
    gk_sigma : float, optional
               Sigma value for Gaussian kernel
    out : numpy.ndarray, optional
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    threshold : float
                Threshold used to create binary mask
    '''
    
    if img.ndim != 2 and img.ndim != 3: raise ValueError, "Requires 2 or 3 dimensional images"
    _logger.debug("Tight mask - started")
    if threshold is None or threshold == 'A': 
        _logger.debug("Finding threshold")
        threshold = analysis.otsu(img.ravel())
    else: threshold=float(threshold)
    
    _logger.debug("Finding biggest object")
    if out is None: out = numpy.empty_like(img)
    out[:]=0
    out[biggest_object_select(img>threshold)]=1.0
    
    _logger.debug("Dilating")
    # Dilate the binary image
    if ndilate > 0:
        elem = scipy.ndimage.generate_binary_structure(out.ndim, 2)
        out[:]=scipy.ndimage.binary_dilation(out, elem, ndilate)
    
    
    if gk_size > 0 and gk_sigma > 0:
        _logger.debug("Smoothing in real space")
        out = gaussian_smooth(out, gk_size, gk_sigma, out)
    _logger.debug("Tight mask - finished")
    return out, threshold

def gaussian_smooth(img, gk_size=3, gk_sigma=3.0, out=None):#, mode='reflect', cval=0.0):
    ''' Perform real space smoothing with a Gaussian kernel on an image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    gk_size : int, optional
              Size of Gaussian kernel used for real space smoothing
    gk_sigma : float, optional
               Sigma value for Gaussian kernel
    out : numpy.ndarray, optional
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if out is None: out = numpy.empty_like(img)
    # Smooth the image with a Gausian kernel of size `kernel_size`, and smoothness `gauss_standard_dev`
    #return scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect', output=out)
    K = gaussian_kernel(tuple([gk_size for i in xrange(img.ndim)]), gk_sigma)
    
    K  /= (K.mean()*numpy.prod(K.shape))
    if img.ndim == 2:
        out[:] = scipy.signal.convolve2d(img, K, mode='same', boundary='wrap') #symm
    else:
        out[:]=scipy.ndimage.convolve(img, K, mode='mirror')#, mode=mode, cval=cval)#, mode='mirror')
    return out

def dialate_mask(img, ndialate, out=None):
    '''
    '''
    
    if out is None: out = numpy.empty_like(img)
    elem = scipy.ndimage.generate_binary_structure(out.ndim, 2)
    out[:]=scipy.ndimage.binary_dilation(out, elem, ndialate)
    return out

def gaussian_kernel(shape, sigma, dtype=numpy.float, out=None):
    ''' Create a centered Gaussian kernel
    
    :Parameters:
    
    shape : tuple
            Shape of the kernel
    sigma : float
            Width of the Guassian
    dtype : numpy.dtype
            Data type
    out : numpy.ndarray, optional
          Output image
    
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    #from ..util.numpy_ext import meshgrid
    shape = numpy.asarray(shape)
    center = numpy.asarray(shape, dtype=numpy.int)//2
    sigma2 = 2*sigma*sigma
    sr2pi = numpy.sqrt(2.0*numpy.pi)
    rng=[]
    norm = 1.0
    for s,c in zip(shape,center): 
        rng.append(numpy.arange(0, s, dtype=numpy.float)-c)
        norm *= 1.0/(sigma*sr2pi)
    rng = numpy.meshgrid(*rng, indexing='xy')
    val = (rng[0].astype(numpy.float)**2)/sigma2
    for i in xrange(1, len(rng)):
        val += (rng[i].astype(numpy.float)**2)/sigma2
    out = numpy.exp(-val, out)
    return numpy.multiply(out, norm, out)

def biggest_object(img, out=None):
    ''' Get the biggest object in a binary image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    threshold : float, optional
                Threshold for binarization, if not specified us Otsu's method to find
    out : numpy.ndarray, optional
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if img.dtype != numpy.bool: raise ValueError, "Requires binary image: %s"%img.__class__.__name__
    elem = None #numpy.ones((3,3)) if img.ndim == 2 else numpy.ones((3,3,3))
    label, num_label = scipy.ndimage.label(img, elem)
    biggest = numpy.argmax(numpy.histogram(label, num_label+1)[0][1:])+1
    #biggest = numpy.argmax([numpy.sum(l==label) for l in xrange(1, num_label+1)])+1
    #assert(biggest==biggest1)
    if out is None: out = numpy.zeros(img.shape)#, dtype=img.dtype)
    out[label == biggest] = 1
    return out

def histeq(img, hist_bins=256, **extra):
    ''' Equalize the histogram of an image
    
    :Parameters:
    
    img : array
          Image data
    hist_bins : int
                Number of bins for the histogram
    
    :Returns:
    
    img : array
          Histogram equalized image
          
    .. note::
    
        http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    '''
    
    imhist,bins = numpy.histogram(img.flatten(),hist_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = numpy.interp(img.flatten(), bins[:-1], cdf)#use linear interpolation of cdf to find new pixel values
    img = im2.reshape(img.shape)
    return img

def hist_match(img, ref, hist_bins=256, **extra):
    ''' Equalize the histogram of an image
    
    :Parameters:
    
    img : array
          Image data
    hist_bins : int
                Number of bins for the histogram
    
    :Returns:
    
    img : array
          Histogram equalized image
          
    .. note::
    
        http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    '''
    
    imhist,bins = numpy.histogram(ref.flatten(),hist_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = numpy.interp(img.flatten(), bins[:-1], cdf)#use linear interpolation of cdf to find new pixel values
    img = im2.reshape(img.shape)
    return img

def biggest_object_select(img):
    ''' Get the biggest object in a binary image
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    threshold : float, optional
                Threshold for binarization, if not specified us Otsu's method to find
    out : numpy.ndarray, optional
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if img.dtype != numpy.bool: raise ValueError, "Requires binary image: %s"%img.__class__.__name__
    elem = None #numpy.ones((3,3)) if img.ndim == 2 else numpy.ones((3,3,3))
    label, num_label = scipy.ndimage.label(img, elem)
    #biggest = numpy.argmax(numpy.histogram(label, num_label+1)[0][1:])+1
    tmp = numpy.histogram(label, num_label+1)[0][1:]
    biggest = numpy.argmax(tmp)
    #_logger.info("biggest: %f -> %f | %d, %d"%(tmp[biggest]/float(tmp.shape[0]), numpy.max(tmp), numpy.sum(label == biggest), numpy.sum(label == (biggest+1))))
    biggest += 1
    return label == biggest

def bispectrum(signal, maxlag=0.0081, window='uniform', scale='unbiased'):
    ''' Compute the bispectrum of a 1 or 2 dimensional array
    
    :Parameters:
    
    signal : array
             Input array
    maxlag : int
             Maximum bispectrum lag (<= signal length)
    window : string
            Window mode :
             - 'none' does not compute a window
             - 'uniform' computes the uniform hexagonal window
             - 'sasaki' computes the sasaki window
             - 'priestley' computes the priestley window
             - 'parzen' computes the parzen window
             - 'hamming' computes the hamming window
             - 'gaussian' computes the gaussian distribution window
             - 'daniell' computes the daniell window
    scale : string
            Scale mode :
             - 'biased' computes the biased estimate
             - 'unbiased' computes the unbiased estimate
    
    :Returns:

    out : array
          Output matrix
    '''
        
    # Compute lag vector
    sr = numpy.shape(signal)
    sample = sr[0]
    if sample == 1:
        signal = signal.T
        sample = sr[1]
        record = 1
    else:
        record = sr[1] 
  
    # print errors
    if maxlag >= sample:
        raise ValueError('Maxlag must be an integer smaller than the signal vector') 
    if numpy.logical_and(numpy.logical_and(scale != 'u',scale != 'b'),numpy.logical_and(scale != 'unbiased',scale != 'biased')):
        raise ValueError('Scale must be either biased, b, unbiased or u') 
    
    freq = numpy.arange(-maxlag,maxlag, dtype=numpy.float)/maxlag/2
    
    # Generate Constants
    maxlag1 = maxlag+1
    maxlag2 = maxlag*2
    maxlag21 = maxlag2+1
    samp1ind = numpy.arange(sample,0,-1, dtype=numpy.int)
    samlsamind = numpy.arange(sample-maxlag,sample+1, dtype=numpy.int)
    ml1samind = numpy.arange(maxlag1,sample+1, dtype=numpy.int)
    ml211ind = numpy.arange(maxlag21,0,-1, dtype=numpy.int)
    zeros1maxlag = numpy.zeros(maxlag)
    zerosmaxlag1 = zeros1maxlag
    onesmaxlag211 = numpy.ones(maxlag21,dtype=int)
    
    # Subtract mean from signal 
    if record > 1:
        ave = numpy.mean(signal.T,axis = 1)
        meansig = numpy.zeros([sample,record])
        for n in numpy.arange(0,sample):
            meansig[n,:] = ave
    else:   
        meansig = numpy.mean(signal)
    signal = signal-meansig

    # Prepare cumulant matrix
    cum = numpy.zeros([maxlag21,maxlag21])
    for k in numpy.arange(0,record):
        sig = signal[:,k]
        sig = sig.reshape((len(sig)),1)
        trflsig=sig[samp1ind-1].T
        trflsig = trflsig[0] #??
        toepsig = scipy.linalg.toeplitz(numpy.hstack([numpy.ravel(sig[samlsamind-1]),numpy.ravel(zerosmaxlag1)]),numpy.hstack([numpy.ravel(numpy.conj(trflsig[ml1samind-1])),zeros1maxlag]))
        t = numpy.zeros([len(onesmaxlag211),len(trflsig)])
        for n in numpy.arange(0,len(onesmaxlag211)):
            t[n,:] = trflsig
        cum = cum + numpy.dot(toepsig*t,toepsig.T)
    cum = cum/record
    if numpy.logical_or(scale == 'b',scale == 'biased'):
        cum = cum/sample
    else:
        scalmat=numpy.zeros([maxlag1,maxlag1])
        for k in numpy.arange(0,maxlag1):
            maxlag1k = maxlag1-k-1
            scalmat[k,k:maxlag1] = numpy.tile(sample-maxlag1k,(1,maxlag1k+1))
        scalmat = scalmat + numpy.triu(scalmat,1).T
        samplemaxlag1 = sample-maxlag1
        maxlag1ind = numpy.arange(maxlag,0,-1)
        a = scipy.linalg.toeplitz(numpy.arange(samplemaxlag1,sample-1), numpy.arange(samplemaxlag1,sample-maxlag2-1,-1))
        a = numpy.vstack([a,scalmat[maxlag1-1,maxlag1ind-1]])
        scalmat = numpy.hstack([scalmat,a])
        scalmat = numpy.vstack([scalmat,scalmat[numpy.ix_(maxlag1ind-1,ml211ind-1)]])
        [r,c] = numpy.nonzero(scalmat<1)
        scalmat[r,c] = 1
        cum = cum/scalmat    
    wind = lagwind(maxlag1,window);
    we = numpy.ravel(numpy.hstack([wind[numpy.arange(maxlag1-1,0,-1)], wind]))
    windeven = numpy.zeros([len(onesmaxlag211),len(we)]) 
    for n in numpy.arange(0,len(onesmaxlag211)):
        windeven[n,:] = we
    wind = numpy.hstack([wind,zeros1maxlag])
    wind = scipy.linalg.toeplitz([wind,numpy.hstack([wind[0],numpy.zeros(maxlag2)])])
    wind = scipy.tril(wind[0:maxlag21,0:maxlag21])
    wind = wind + scipy.tril(wind,-1).T
    wind = wind[ml211ind-1,:]*windeven*windeven.T
    bisp = scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.ifftshift(cum*wind)))
    a = numpy.fft.ifftshift(cum*wind)
    return bisp, freq

def lagwind(lag,window):
    ''' Compute the bispectrum of a 1 or 2 dimensional array
    
    TODO: fix: none, parzen
    
    :Parameters:
    
    maxlag : int
            lagging term
    window : string
            'none' does not compute a window
            'uniform' computes the uniform hexagonal window
            'sasaki' computes the sasaki window
            'priestley' computes the priestley window
            'parzen' computes the parzen window
            'hamming' computes the hamming window
            'gaussian' computes the gaussian distribution window
            'daniell' computes the daniell window

    :Returns:

    out : array
            Output window array
    '''
    
    lag = float(lag)
    lag1 = lag-1
    if lag == 1:
        return 1
    windows = ['uniform','sasaki','priestley', 'parzen', 'hamming', 'gaussian', 'daniell'];
    try: int(window)
    except: i = windows.index(window)+1
    else: i = window+1
    if i == 1:
        wind = numpy.ones(lag)
    elif i == 2:
        windlag = numpy.arange(lag)/lag1
        wind=numpy.sin(numpy.pi*windlag)/numpy.pi+numpy.cos(numpy.pi*windlag)*(1-windlag)
    elif i == 3:
        windlag = (numpy.arange(float(lag1))+1)/lag1
        w = (numpy.sin(numpy.pi*windlag)/numpy.pi/windlag-numpy.cos(numpy.pi*windlag))*3/numpy.pi/numpy.pi/windlag/windlag
        wind = numpy.ones(lag)
        wind[1:len(wind)] = w
    elif i == 4:
        fixlag121 = int(lag1/2)+1
        windlag0 = numpy.arange(fixlag121+1)/lag1
        windlag1 = 1-numpy.arange(fixlag121,lag1+1)/lag1
        wind1 = 1-(1-windlag0)*windlag0*windlag0*6
        wind2 = windlag1*windlag1*windlag1*2
        wind = numpy.ones(len(wind1)+len(wind2))
        wind[0:len(wind1)] = wind1
        wind[len(wind1):len(wind)] = wind2
    elif i == 5:
        wind=0.54+0.46*numpy.cos(numpy.pi*numpy.arange(lag1+1)/lag1)
    elif i == 6:
        w1 = scipy.special.erfc(((numpy.arange(1,lag1)/(lag1)-.5)*8)/numpy.sqrt(2))/2
        w2 = numpy.zeros(len(w1)+1)
        w2[0:len(w2)-1] = w1
        wind = numpy.ones(len(w2)+1)
        wind[1:len(wind)] = w2
    elif i == 7:
        windlag = numpy.arange(1,lag)/lag1
        w1 = numpy.sin(numpy.pi*windlag)/numpy.pi/windlag
        wind = numpy.ones(len(w1)+1)
        wind[1:len(wind)] = w1
    return wind

def major_axis_angle(ellipse):
    ''' Calculate the major axis angle
    '''
    
    _logger.error("%s"%(str(ellipse)))
    a, b, c, _, _, _ = ellipse
    
    _logger.error("%s"%(str(a)))
    
    if b == 0:
        if a < c: return 0
        if a > c: return 0.5*numpy.pi
    else:
        if a < c: return 0.5*numpy.arctan(2*b/(a-c))
        if a > c: return 0.5*numpy.pi + 0.5*numpy.arctan(2*b/(a-c))
    raise ValueError, "Invalid ellipse parameters: %f, %f"%(a, c)

def fit_ellipse(x, y):
    '''fit an ellipse to the points.
    
    INPUT: 
      xy -- N x 2 -- points in 2D
      
    OUTPUT:
      A,B,C,D,E,F -- real numbers such that:
        A * x**2 + B * x * y + C * y**2 + D * x + E * y + F =~= 0
    
    This is an implementation of:
    
    "Direct Least Square Fitting of Ellipses"
    by Andrew Fitzgibbon, Maurizio Pilu, and Robert B. Fisher
    IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 
    VOL. 21, NO. 5, MAY 1999
    
    Shai Revzen, U Penn, 2010  
    '''
    
    D = numpy.c_[ x*x, x*y, y*y, x, y, numpy.ones_like(x) ]
    S = numpy.dot(D.T,D)
    C = numpy.zeros((6,6),D.dtype)
    C[0,2]=-2
    C[1,1]=1
    C[2,0]=-2
    geval,gevec = scipy.linalg.eig( S, C )
    idx = numpy.nonzero( geval<0 & ~ numpy.isinf(geval) )
    _logger.info("idx: %s"%str(idx))
    _logger.info("geval: %s"%str(geval))
    _logger.info("gevec: %s"%str(gevec))
    return tuple(gevec[:,idx].real)

def fourier_shell_correlation(img1, img2, center=None, pad=1, out=None):
    ''' Estimate the resolution using the Fourier Shell/Ring correlation. 
    Works for both 2D and 3D images.
    
    :Parameters:
    
    img1 : array
           Image data in 2D or 3D
    img2 : array
           Image data in 2D or 3D
    
    :Returns:
    
    out : array
          1d radial profile
    '''
    
    if img1.dim != img2.dim: raise ValueError, "Dimensions of array must match"
    if img1.dim not in (2,3): raise ValueError, "Must be either 2 or 3D array"
    if img1.shape != img2.shape: raise ValueError, "Shape of both arrays must match"
    if pad > 1: 
        img1 = pad_image(img1.astype(numpy.complex64), (int(img1.shape[0]*pad), int(img1.shape[1]*pad)), 'm')
        img2 = pad_image(img2.astype(numpy.complex64), (int(img2.shape[0]*pad), int(img2.shape[1]*pad)), 'm')
    if out is None: out = numpy.zeros(img1.shape)
    
    if img1.ndim == 2:
        fimg1 = scipy.fftpack.fftshift(scipy.fftpack.fft2(img1))
        fimg2 = scipy.fftpack.fftshift(scipy.fftpack.fft2(img2))
    else:
        fimg1 = scipy.fftpack.fftshift(scipy.fftpack.fftn(img1))
        fimg2 = scipy.fftpack.fftshift(scipy.fftpack.fftn(img2))
    
        out[:] = numpy.multiply(fimg1, fimg2.conjugate()).real
        numpy.abs(fimg1, fimg1)
        numpy.square(fimg1.real, fimg1.real)
        numpy.abs(fimg2, fimg2)
        numpy.square(fimg2.real, fimg2.real)
        numpy.multiply(fimg1.real, fimg2.real, fimg1.real)
        numpy.sqrt(fimg1.real, fimg1.real)
        numpy.divide(out, fimg1.real)
    if img1.ndim == 2:
        return mean_azimuthal(out, center)
    else:
        return mean_azimuthal_3d(out, center)

def sum_by_group(values, groups):
    ''' Sum values by group labels
    
    :Parameters:
    
    values : array
             Array of values to sum
    groups : array
             Array containing group membership for each value in `values` array
             
    :Returns:
    
    out : array
          Size corresponding to the number of groups where each element
          is the sum values in that group
    
    .. note::
    
        Taken from ResMap:  http://sourceforge.net/projects/resmap/
        Originally from http://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy
    '''
    
    order             = numpy.argsort(groups)
    groups            = groups[order]
    values            = values[order]
    values.cumsum(out =values)
    index             = numpy.ones(len(groups), dtype=numpy.bool)
    index[:-1]        = groups[1:] != groups[:-1]
    values            = values[index]
    groups            = groups[index]
    values[1:]        = values[1:] - values[:-1]
    return values

def mean_azimuthal_3d(image, center=None, binsize=1.0):
    '''Calculate the spherically averaged profile.

    :Parameters:
    
    image : array
            3D image
    center : tuple
             The [x,y,z] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    binsize : float
              Size of the averaging bin.  Can lead to strange results if
              non-binsize factors are used to specify the center and the binsize is too large

    :Returns:
    
    out : array
          1D array containing mean radial average
    
    .. note::
        
        Taken from ResMap: http://sourceforge.net/projects/resmap/
    
    '''

    n         = image.shape[0]

    [x,y,z] = numpy.mgrid[ -n/2:n/2:numpy.complex(0,n),
                        -n/2:n/2:numpy.complex(0,n),
                        -n/2:n/2:numpy.complex(0,n) ]
    r       = numpy.array(numpy.sqrt(x**2 + y**2 + z**2), dtype=numpy.float32)

    nbins  = int(numpy.round( ((n/2.0) - 1) / binsize))
    maxbin = nbins * binsize
    bins   = numpy.linspace(0,maxbin,nbins)

    # Find out which radial bin each point in the map belongs to
    whichbin = numpy.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = numpy.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by xrange(1,nbins+1) )
    radial_prof = sum_by_group(image.flat,whichbin)/nr

    return radial_prof



