''' Image enhancement functions

This module contains a set of image enhancement functions using NumPy, SciPy and 
custom wrapped fortran functions (taken from SPIDER).

.. todo:: automatic inversion detection? for defocus? based on ctf

.. Created on Jul 31, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
from eman2_utility import em2numpy2em as _em2numpy2em, em2numpy2res as _em2numpy2res
import analysis
import numpy, scipy, math, logging, scipy.ndimage
import scipy.fftpack, scipy.signal
import scipy.ndimage.filters
import scipy.ndimage.morphology
'''
try: 
    from scipy.signal import find_peaks_cwt
    find_peaks_cwt;
except: from util._peak_finding import find_peaks_cwt
'''
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
try: 
    from util import _spider_util
    _spider_util;
except:
    if _logger.isEnabledFor(logging.DEBUG):
        tracing.log_import_error('Failed to load _spider_util.so module - certain functions will not be available: ndimage_utility.ramp', _logger)
    try:
        import _spider_util
        _spider_util;
    except:
        tracing.log_import_error('Failed to load _spider_util.so module - certain functions will not be available: ndimage_utility.ramp', _logger)
"""
@_em2numpy2res
def find_peaks(cc, width):
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
    
    peaks = find_peaks_cwt(cc.ravel(), numpy.asarray([width]))
    ccv = cc[peaks].copy().squeeze()
    peaks = numpy.unravel_index(peaks, cc.shape)
    return numpy.hstack((ccv[:, numpy.newaxis], peaks))
"""

def mean_azimuthal(img, center=None):
    ''' Calculate the sum of a 2D array along the azimuthal
    
    :Parameters:
    
    img : array-like
          Image array
    center : tuple, optional
              Coordaintes of the image center
    
    :Returns:
    
    out : array
          Sum over the radial lines of the image
    
    .. note::
    
        Adopted from https://github.com/numpy/numpy/pull/230/files#r851142
    '''
    
    img = numpy.asanyarray(img)
    if img.ndim != 2: raise ValueError, "Input array must be 2D"
    if center is None: center = (img.shape[0]/2, img.shape[1]/2)
    i, j = numpy.arange(img.shape[0])[:, None], numpy.arange(img.shape[1])[None, :]   
    i, j = i-center[0], j-center[1]
    k = (j**2+i**2)**.5
    k = k.astype(int)
    return numpy.bincount(k.ravel(), img.ravel())/numpy.bincount(k.ravel())

@_em2numpy2res
def find_peaks_fast(cc, width):
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
    if center is None:
        cx, cy = shape[0]/2, shape[1]/2
    elif not hasattr(center, '__iter__'):
        cx, cy = center, center
    else:
        cx, cy = center
    radius2 = radius+1
    y, x = numpy.ogrid[-radius2: radius2, -radius2: radius2]
    index = x**2 + y**2 <= radius**2
    a[cy-radius2:cy+radius2, cx-radius2:cx+radius2][index] = 1
    return a

@_em2numpy2em
def ramp(img, out=None):
    '''Remove change in illumination across an image
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:
    
    img : numpy.ndarray
          Ramped image
    '''
    
    if out is None: out = img.copy()
    if _spider_util is None: raise ImportError, 'Failed to load _spider_util.so module - function `ramp` is unavailable'
    _spider_util.ramp(out.T)
    return out

def cross_correlate_raw(img, template, out=None):
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
    out[:,:] = scipy.fftpack.ifft2(fp1).real
    return out

def cross_correlate(img, template, out=None):
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
    
    return scipy.fftpack.fftshift(cross_correlate_raw(img, template, out))

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
    
    .. note::
    
        Taken from: https://github.com/numpy/numpy/pull/31
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
        raise ValueError("`window` length must be less or equal `array` dimension.") 

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
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

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


def powerspec_avg(imgs, pad):
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
    for img in imgs:
        pad_width = img.shape[0]*pad
        fimg = scipy.fftpack.fft2(pad_image(img, (pad_width, pad_width)))
        if avg is None: avg = fimg.copy()
        else: avg += fimg
    return scipy.fftpack.fftshift(avg).real

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
    
    if out is None: 
        if img.shape[0] == shape[0] and img.shape[1] == shape[1]: return img
        out = numpy.zeros(shape)
        if fill != 0: out[:, :] = fill
    cx = (shape[0]-img.shape[0])/2
    cy = (shape[1]-img.shape[1])/2
    out[cx:cx+img.shape[0], cy:cy+img.shape[1]] = img
    return out

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
    return scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect', output=out)

def filter_butterworth_lp(img, low_cutoff, high_cutoff):
    '''
    
    ..todo:: finish this function
    
    .. note:: 
        
        Taken from:
        http://www.psychopy.org/epydoc/psychopy.filters-pysrc.html
        and
        Sparx
    
    '''
    
    eps = 0.882
    aa = 10.624
    n = 2 * numpy.log10(eps/numpy.sqrt(aa*aa-1.0))/numpy.log10(low_cutoff/high_cutoff)
    cutoff = low_cutoff/numpy.power(eps, 2.0/n)
    #todo: pad
    x =  numpy.linspace(-0.5, 0.5, img.shape[1]) 
    y =  numpy.linspace(-0.5, 0.5, img.shape[0])
    radius = numpy.sqrt((x**2)[numpy.newaxis] + (y**2)[:, numpy.newaxis])
    f = 1 / (1.0 + (radius / cutoff)**n);
    return filter_image(img, f)

def filter_image(img, filt, pad=1):
    '''
    .. todo:: filter padding
    '''
    
    #fshape = numpy.asarray(img.shape)*pad
    fimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(img))
    numpy.multiply(fimg, filt, fimg)
    return scipy.fftpack.ifft2(scipy.fftpack.ifftshift(fimg)).real

@_em2numpy2em
def histogram_match(img, mask, ref, bins=0, iter_max=100, out=None):
    '''Remove change in illumination across an image
    
    :Parameters:

    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Mask for the image
    ref : numpy.ndarray
          Reference noise image
    bins : int, optional
           Number of bins for the histogram, if 0 choose image_size/16
    iter_max : int, optional
               Number of iterations for downhill simplex
    out : numpy.ndarray
          Output image
    
    :Returns:
    
    img : numpy.ndarray
          Enhanced image
    '''
    
    if out is None: out = img.copy()
    if _spider_util is None: raise ImportError, 'Failed to load _spider_util.so module - function `histogram_match` is unavailable'
    if mask.dtype != numpy.bool: mask = mask>0.5
    if bins == 0: bins = len(out.ravel())/16
    _spider_util.histc2(out.ravel(), mask.ravel(), ref.ravel(), int(bins), int(iter_max))
    return out

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

@_em2numpy2em
def fftamp(img, out=None):
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
    
    fimg = numpy.fft.fft2(img)
    fimg = numpy.fft.fftshift(fimg)
    fimg = fimg*fimg.conjugate()
    out = numpy.abs(fimg, out)
    return out

def logpolar(image, angles=None, radii=None, out=None):
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
    center = shape[0] / 2, shape[1] / 2
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
def segment(img, bins=0, out=None):
    ''' Segment the given image with Otsu's method
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    bins : int
           Number of bins to use in discretizing data for Otsu's method
    out : numpy.ndarray
          Output image
        
    :Returns:
    
    out : numpy.ndarray
          Normalized image
    '''
    
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
    
    mdata = img*mask if mask is not None else img
    out = numpy.subtract(img, numpy.mean(mdata), out)
    if var_one:
        numpy.divide(out, numpy.std(mdata), out)
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
    
    vmin = numpy.min(img) if mask is None else numpy.min(img*mask)
    out = numpy.subtract(img, vmin, out)
    vmax = numpy.max(img) if mask is None else numpy.max(img*mask)
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
def replace_outlier(img, dust_sigma, xray_sigma=0, out=None):
    '''Clamp outlier pixels, either too black due to dust or too white due to hot-pixels. Replace with 
    samples drawn from the normal distribution with the same mean and standard deviation.
    
    :Parameters:

    img : numpy.ndarray
          Input image
    dust_sigma : float
                 Number of standard deviations for black pixels
    xray_sigma : float
                 Number of standard deviations for white pixels
    out : numpy.ndarray
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if out is None: out = img.copy()
    avg = numpy.mean(img)
    std = numpy.std(img)
    vmin = numpy.min(img)
    vmax = numpy.max(img)
    
    avg1, std1, vmin1, vmax1 = avg, std, vmin, vmax
    start = int( max((vmax-avg)/std, (avg-vmin)/std) )
    for nstd in xrange(start, int(min(abs(dust_sigma), xray_sigma))-1, -1):
        hcut = avg+std*nstd
        lcut = avg-std*nstd
        sel = numpy.logical_and(avg > lcut, avg < hcut)
        avg = numpy.mean(img[sel])
        std = numpy.std(img[sel])
    
    if dust_sigma > 0: dust_sigma = -dust_sigma
    if xray_sigma == 0: xray_sigma=dust_sigma
    if ((vmin1 - avg1) / std1) < dust_sigma:
        sel = ((img - avg)/std) < dust_sigma
        out[sel] = numpy.random.normal(avg, std, numpy.sum(sel))
    if ((vmax1 - avg1) / std1) > xray_sigma:
        sel = ((img - avg)/std) > xray_sigma
        out[sel] = numpy.random.normal(avg, std, numpy.sum(sel))
    
    return out

@_em2numpy2em
def crop_window(img, x, y, offset, out=None):
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
    
    xb = x-offset
    yb = y-offset
    width = offset*2
    xe = xb+width
    ye = yb+width
    
    if out is None: out = numpy.zeros((width, width), dtype=img.dtype)
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
        out[dyb:width-dye, dxb:width-dxe] = img[yb:ye, xb:xe]
    except:
        _logger.error("Error in window1 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dxe > 0: out[dyb:width-dye, width-dxe:] = img[yb:ye, xb-dxe:xb]
    except:
        _logger.error("Error in window2 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
        raise
    try:
        if dye > 0: out[width-dye:, dxb:width-dxe] = img[yb-dye:yb, xb:xe]
    except:
        _logger.error("Error in window3 %d,%d - %d,%d | %d,%d - %d,%d"%(xb, xe, yb, ye, dxb, dxe, dyb, dye))
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
    '''
    
    if img.ndim != 2 and img.ndim != 3: raise ValueError, "Requires 2 or 3 dimensional images"
    if threshold is None: threshold = analysis.otsu(img.ravel())
    
    # Get largest component in the binary image
    out = biggest_object(img>threshold, out)
    
    # Dilate the binary image
    elem = scipy.ndimage.generate_binary_structure(out.ndim, 2)
    out[:]=scipy.ndimage.binary_dilation(out, elem, ndilate)
    
    # Smooth the image with a Gausian kernel of size `kernel_size`, and smoothness `gauss_standard_dev`
    #return scipy.ndimage.filters.gaussian_filter(out, sigma, mode='reflect', output=out)
    
    if gk_size > 0 and gk_sigma > 0:
        K = gaussian_kernel(tuple([gk_size for i in xrange(img.ndim)]), gk_sigma)
        K /= (numpy.mean(K)*numpy.prod(K.shape))
        out[:] = scipy.ndimage.convolve(out, K)
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
    
    from util._new_numpy import meshgrid
    shape = numpy.asarray(shape)/2
    sigma2 = 2*sigma*sigma
    sr2pi = numpy.sqrt(2.0*numpy.pi)
    rng=[]
    for s in shape: rng.append(numpy.arange(-s, s+1, dtype=numpy.float))
    rng = meshgrid(*rng, indexing='xy')
    val = (rng[0].astype(numpy.float)**2)/sigma2
    norm = 1.0/(sigma*sr2pi)
    for i in xrange(1, len(rng)):
        val += (rng[i].astype(numpy.float)**2)/sigma2
        norm *= 1.0/(sigma*sr2pi)
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
    
    if img.dtype != numpy.bool: raise ValueError, "Requires binary image"
    elem = None #numpy.ones((3,3)) if img.ndim == 2 else numpy.ones((3,3,3))
    label, num_label = scipy.ndimage.label(img, elem)
    biggest = numpy.argmax([numpy.sum(l==label) for l in xrange(1, num_label+1)])+1
    if out is None: out = numpy.zeros(img.shape)#, dtype=img.dtype)
    out[label == biggest] = 1
    return out

