''' Filter 2D or 3D images

This module contains both linear Fourier-accelerated filters as well as ramp 
correction and histogram equalization. The filtering code is performed
using Fortran accelerated code from the SPIDER library.

SPIDER: http://www.wadsworth.org/spider_doc/spider/docs/spider.html 

.. Created on Jun 30, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy
import scipy.fftpack

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from spi import _spider_filter
    _spider_filter;
except:
    _spider_filter=None
    tracing.log_import_error('Failed to load _spider_filter.so module', _logger)
    
def histogram_match(img, mask, ref, bins=0, iter_max=500, out=None):
    ''' Adjust the image intensity to match a standard reference, e.g. noise image
    
    This function adjusts the histogram of the input image to match that of a reference
    image under a given mask. The reference is generally a noise image.
    
    .. note::
        
        For more detail, see: http://www.wadsworth.org/spider_doc/spider/docs/man/cefit.html
    
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
    
    img = img.astype(numpy.float32)
    ref = ref.astype(numpy.float32)
    if out is None: out = img.copy()
    if mask.dtype != numpy.bool: mask = mask>0.5
    if bins == 0: bins = min(len(out.ravel())/16, 256)
    if out.ravel().shape[0] != mask.ravel().shape[0]: raise ValueError, "Image shape != mask shape"
    if out.ravel().shape[0] != ref.ravel().shape[0]: raise ValueError, "Image shape != reference shape"
    _spider_filter.histeq(out.ravel(), mask.ravel(), ref.ravel(), int(bins), int(iter_max))
    return out

def ramp(img, out=None):
    '''Remove change in illumination across an image
    
    This function removes a wedge-like profile from the image
    
    .. note::
        
        For more detail, see: http://www.wadsworth.org/spider_doc/spider/docs/man/ra.html
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:
    
    img : numpy.ndarray
          Ramped image
    '''
    
    if out is None: 
        out = numpy.empty_like(img, dtype=numpy.float32)
    elif out.dtype != numpy.float32: raise ValueError, "Output array must be float 32"
    if numpy.byte_bounds(out) != numpy.byte_bounds(img):  out[:]=img
    tout = out.T if not out.flags.f_contiguous else out
    if _spider_filter.ramp(tout) != 0:
        raise ValueError, "Ramp filter failed"
    return out

def _modnx(s): return s+2-numpy.mod(s, 2)

def filter_gaussian_lowpass(img, sigma, pad=2):
    ''' Filter an image with the SPIDER Gaussian low pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    sigma : float
            Frequency cutoff
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.gaussian_lp_3d(out.T, float(sigma), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the gaussian filter"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.gaussian_lp_2d(out.T, float(sigma), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the gaussian filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def filter_gaussian_highpass(img, sigma, pad=2):
    ''' Filter an image with the SPIDER Gaussian high pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    sigma : float
            Frequency cutoff
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.gaussian_hp_3d(out.T, float(sigma), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.gaussian_hp_2d(out.T, float(sigma), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the gaussian filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def filter_butterworth_lowpass(img, lcut, hcut, pad=2):
    ''' Filter an image with the SPIDER Butterworth low pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    lcut : float
           Frequency lower limit
    hcut : float
           Frequency upper limit
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.butter_lp_3d(out.T, float(lcut), float(hcut), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.butter_lp_2d(out.T, float(lcut), float(hcut), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the butter filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def filter_butterworth_highpass(img, lcut, hcut, pad=2):
    ''' Filter an image with the SPIDER Butterworth high pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    lcut : float
           Frequency lower limit
    hcut : float
           Frequency upper limit
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.butter_hp_3d(out.T, float(lcut), float(hcut), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.butter_hp_2d(out.T, float(lcut), float(hcut), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the butter filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def filter_raised_cosine_lowpass(img, lcut, hcut, pad=2):
    ''' Filter an image with the SPIDER Butterworth low pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    lcut : float
           Frequency lower limit
    hcut : float
           Frequency upper limit
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.rcos_lp_3d(out.T, float(lcut), float(hcut), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.rcos_lp_2d(out.T, float(lcut), float(hcut), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the rcos filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def filter_raised_cosine_highpass(img, lcut, hcut, pad=2):
    ''' Filter an image with the SPIDER Butterworth high pass filter
    
    .. note::
    
        http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html
    
    :Parameters:
    
    img : array
          Image to filter 2D or 3D
    lcut : float
           Frequency lower limit
    hcut : float
           Frequency upper limit
    pad : int
          Number of times to pad image
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    if pad < 1: pad =1
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Supports only 2D or 3D images"
    if img.ndim == 3:
        nxp = int(img.shape[2]*pad)
        out = numpy.empty((int(img.shape[0]*pad), int(img.shape[1]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        if _spider_filter.rcos_hp_3d(out.T, float(lcut), float(hcut), nxp, img.shape[2], img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred"
        out = out[:img.shape[0], :img.shape[1], :img.shape[2]]
    else:
        nxp = int(img.shape[1]*pad)
        out = numpy.empty((int(img.shape[0]*pad), _modnx(nxp)), dtype=numpy.float32)
        out[:img.shape[0], :img.shape[1]] = img
        if _spider_filter.rcos_hp_2d(out.T, float(lcut), float(hcut), nxp, img.shape[1], img.shape[0]) != 0:
            raise StandardError, "An unknown error occurred in the rcos filter"
        out = out[:img.shape[0], :img.shape[1]]
    return out

def gaussian_lowpass_kernel(shape, low_cutoff, dtype):
    ''' Create a Gaussian high pass kernel of the given
    shape.
    
    .. note:: 
        
        Implementation follows from SPARX
    
    :Parameters:
    
    shape : tuple
            Tuple of ints describing the shape of the kernel
    low_cutoff : float
                 Cutoff frequency for low pass filter
    dtype : dtype
            Data type for kernel
    
    :Returns:
    
    kernel : array
             Array of given shape and type defining a kernel
    '''
    
    omega = 0.5/low_cutoff/low_cutoff
    kernel = numpy.zeros(shape, dtype=dtype)
    x,y = grid_image(shape)
    dx2 = (1.0/shape[0])**2
    dy2 = (1.0/shape[1])**2
    irad = x**2*dx2+y**2*dy2
    kernel[:, :] = numpy.exp(-irad*omega)
    return kernel

def gaussian_lowpass(img, low_cutoff, pad=1):
    ''' Apply a Gaussian highpass filter to an image
    
    .. seealso:: :py:func:`gaussian_highpass_kernel`
    
    :Parameters:
    
    img : array
          Image
    low_cutoff : float
                 Low-frequency cutoff
    pad : int
          Padding
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    ctype = numpy.complex128 if img.dtype is numpy.float64 else numpy.complex64
    if pad > 1:
        shape = img.shape
        img = pad_image(img.astype(ctype), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'e')
    else: img = img.astype(ctype)
    img = filter_image(img, gaussian_lowpass_kernel(img.shape, low_cutoff, img.dtype), pad)
    if pad > 1: img = depad_image(img, shape)
    return img

def gaussian_highpass_kernel(shape, high_cutoff, dtype):
    ''' Create a Gaussian high pass kernel of the given
    shape.
    
    .. note:: 
        
        Implementation follows from SPARX
    
    :Parameters:
    
    shape : tuple
            Tuple of ints describing the shape of the kernel
    low_cutoff : float
                 Cutoff frequency for low pass filter
    dtype : dtype
            Data type for kernel
    
    :Returns:
    
    kernel : array
             Array of given shape and type defining a kernel
    '''
    
    omega = 0.5/high_cutoff/high_cutoff
    kernel = numpy.zeros(shape, dtype=dtype)
    x,y = grid_image(shape)
    dx2 = (1.0/shape[0])**2
    dy2 = (1.0/shape[1])**2
    irad = x**2*dx2+y**2*dy2
    kernel[:, :] = 1.0-numpy.exp(-irad*omega)
    return kernel

def gaussian_highpass(img, high_cutoff, pad=1):
    ''' Apply a Gaussian highpass filter to an image
    
    .. seealso:: :py:func:`gaussian_highpass_kernel`
    
    :Parameters:
    
    img : array
          Image
    high_cutoff : float
                 High-frequency cutoff
    pad : int
          Padding
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    ctype = numpy.complex128 if img.dtype is numpy.float64 else numpy.complex64
    if pad > 1:
        shape = img.shape
        img = pad_image(img.astype(ctype), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'e')
    else: img = img.astype(ctype)
    img = filter_image(img, gaussian_highpass_kernel(img.shape, high_cutoff, img.dtype), pad)
    if pad > 1: img = depad_image(img, shape)
    return img

def filter_annular_bp_kernel(shape, dtype, freq1, freq2):
    ''' Filter an image with a Gaussian low pass filter
    
    Todo: optimize kernel
    
    :Parameters:
    
    shape : tuple
            Tuple of ints describing the shape of the kernel
    dtype : dtype
            Data type of the image
    freq1 : float
            Cutoff frequency
    freq2 : array
            Cutoff frequency
    
    :Returns:
    
    out : array
          Annular BP kernel
    '''
    
    kernel = numpy.zeros(shape, dtype=dtype)
    irad = radial_image(shape)
    val =  (1.0/(1.0+numpy.exp(((numpy.sqrt(irad)-freq1))/(10.0))))*(1.0-(numpy.exp(-irad /(2*freq2*freq2))))
    kernel[:, :].real = val
    kernel[:, :].imag = val
    return kernel

def filter_annular_bp(img, freq1, freq2, pad=1):
    ''' Filter an image with a Gaussian low pass filter
    
    Todo: optimize kernel
    
    :Parameters:
    
    img : array
          Image to filter
    freq1 : float
            Cutoff frequency
    freq2 : array
            Cutoff frequency
    pad : int
          Padding
    
    :Returns:
    
    out : array
          Filtered image
    '''
    
    ctype = numpy.complex128 if img.dtype is numpy.float64 else numpy.complex64
    if pad > 1:
        shape = img.shape
        img = pad_image(img.astype(ctype), (int(img.shape[0]*pad), int(img.shape[1]*pad)), 'e')
    else: img = img.astype(ctype)
    img = filter_image(img, filter_annular_bp_kernel(img.shape, img.dtype, freq1, freq2), pad)
    if pad > 1: img = depad_image(img, shape)
    return img

def filter_image(img, kernel, pad=1):
    '''
    '''
    
    if img.ndim == 2:
        fimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(img))
        numpy.multiply(fimg, kernel, fimg)
        return scipy.fftpack.ifft2(scipy.fftpack.ifftshift(fimg)).real.copy()
    else:
        fimg = scipy.fftpack.fftshift(scipy.fftpack.fftn(img))
        numpy.multiply(fimg, kernel, fimg)
        return scipy.fftpack.ifftn(scipy.fftpack.ifftshift(fimg)).real.copy()

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
            out[:, :] = (img[0, :].sum()+img[:, 0].sum()+img[img.shape[0]-1, :].sum()+img[:, img.shape[1]-1].sum()) / (img.shape[0]*2+img.shape[1]*2 - 4)
        elif fill == 'r': out[:, :] = numpy.random.normal(img.mean(), img.std(), shape)
        elif fill != 0: out[:, :] = fill
    out[cx:cx+img.shape[0], cy:cy+img.shape[1]] = img
    return out

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

def grid_image(shape, center=None):
    '''
    '''
    
    if not hasattr(shape, '__iter__'): shape = (shape, shape)
    if center is None: cx, cy = shape[0]/2, shape[1]/2
    elif not hasattr(center, '__iter__'): cx, cy = center, center
    else: cx, cy = center
    y, x = numpy.ogrid[-cx: shape[0]-cx, -cy: shape[1]-cy]
    return x, y

def radial_image(shape, center=None):
    '''
    '''
    
    x, y = grid_image(shape, center)
    return x**2+y**2


