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
    
    if out is None: out = numpy.empty_like(img, dtype=numpy.float64)
    out[:]=img
    _spider_filter.ramp(out.T)
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

