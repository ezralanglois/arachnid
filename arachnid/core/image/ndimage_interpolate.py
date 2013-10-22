''' Resize images

This module contains functions to resize an image or a volume. It supports both
bilinear and energy conserving interpolations implemented using Fortran 
accelerated code from the SPIDER library.

SPIDER: http://www.wadsworth.org/spider_doc/spider/docs/spider.html 

.. Created on Mar 8, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging, numpy
import ndimage_interpolate as ndinter
import scipy.fftpack
import ndimage_utility

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)



try:
    from util import _resample
    _resample;
except:
    _resample=None
    tracing.log_import_error('Failed to load _resample.so module', _logger)

try:
    from spi import _spider_interpolate
    _spider_interpolate;
except:
    _spider_interpolate=None
    #_logger.addHandler(logging.StreamHandler())
    #_logger.exception("problem")
    tracing.log_import_error('Failed to load _spider_interpolate.so module', _logger)
    
def decimate_sinc_blackman(img, bin_factor, kernel_size=15):
    '''
    '''
    
    shape = numpy.asarray(img.shape)
    shape /= bin_factor
    bin_factor = 1.0/bin_factor
    fc = 0.5*bin_factor
    ksize=1999
    ltab = int(round(ksize/1.25))
    mhalf = kernel_size//2
    fltb = ltab/(mhalf)
    freq = numpy.arange(kernel_size, dtype=numpy.float)/fltb
    freq[0] = 1e-7
    twopi = 2*numpy.pi
    k = numpy.sin(twopi*freq*fc)/freq*( 0.52 - 0.5*numpy.cos(twopi*-(freq-mhalf)/kernel_size) + 0.08*numpy.cos(2*twopi*(freq-mhalf)/kernel_size)  )
    k = numpy.outer(k,k)
    out = scipy.ndimage.filters.convolve(img, k, mode='mirror').copy()
    return scipy.ndimage.zoom(out, bin_factor, order=2, prefilter=False)
    
    
def sincblackman(bin_factor, template_min = 15, kernel_size=2002, dtype=numpy.float32):
    '''
    '''
    
    bin_factor = 1.0/bin_factor
    frequency_cutoff = 0.5*bin_factor
    kernel = numpy.zeros(kernel_size, dtype=dtype)
    _resample.sinc_blackman_kernel(kernel, int(template_min), float(frequency_cutoff))
    return kernel

def downsample(img, kernel, out):
    '''
    '''
    
    scale=None
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): shape = (int(out[0]), int(out[1]), int(out[2])) if img.ndim == 3 else (int(out[0]), int(out[1]))
        else:
            scale = 1.0/out 
            shape = (int(img.shape[0]/out), int(img.shape[1]/out), int(img.shape[2]/out)) if img.ndim == 3 else (int(img.shape[0]/out), int(img.shape[1]/out))
        out = numpy.zeros(shape, dtype=img.dtype)
    else:
        if out.dtype != img.dtype: raise ValueError, "Requires output and input of the same dtype"
    if kernel.dtype != img.dtype: raise ValueError, "Requires kernel and input of the same dtype"
    if scale is None: scale = float(out.shape[0])/float(img.shape[0])
    _resample.downsample(img, out, kernel, scale)
    return out
    
"""

def _zeropad2(x, shape):
    '''Pad a two-dimensional NumPy array with zeros along its borders
    to the specified shape.
    '''
    m, n = x.shape
    p, q = shape
    assert p > m
    assert q > n
    tb = (p - m) / 2
    lr = (q - n) / 2
    xpadded = _zeros(shape, dtype=complex64)
    xpadded[tb:tb + m, lr:lr + n] = x
    return xpadded
"""
def resample_fourier_window(img, out):
    ''' Attempt at fourier resampling
    '''
    
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): 
            shape = tuple([int(o) for o in out])
        else: 
            shape = tuple([int(s/out) for s in img.shape])
        out = numpy.zeros(shape, dtype=img.dtype)
    if out.shape[0] > img.shape[0]: raise ValueError, "Upsamping not currently supported"
    fimg = scipy.fftpack.fftshift(scipy.fftpack.fft2(img))
    fimg = ndimage_utility.crop_window2(fimg, img.shape[0]/2, img.shape[1]/2, (out.shape[0], out.shape[1]))
    out[:] = scipy.fftpack.ifftshift(scipy.fftpack.ifft2(fimg)).real
    return out
    
def interpolate(img, out, method='bilinear'):
    ''' Interpolate the size of the input image
    
    Available methods:
        
        - bilinear: Bilinear interpolation (not suggest for production work)
        - ft: Energy preserving Fourier-based interpolation
        - fs: Fourier-based bicubic/tricubi spline interpolation

    >>> from arachnid.core.image import ndimage_interpolate
    >>> img = numpy.ones((32,32))
    >>> simg = ndimage_interpolate.interpolate(img, (15,15), 'bilinear')
    >>> print simg.shape
    (15,15)
    
    :Parameters:
    
    img : array
          Image
    out : array or float or list
          Output array or float factor or list of dimensions
          
    :Returns:
    
    out : array
          Interpolated image
    '''
    
    if method not in ('bilinear', 'ft', 'fs'):
        raise ValueError, "method argument must be one of the following: bilinear,ft,fs"
    return getattr(ndinter, 'interpolate_'+method)(img, out)

def interpolate_bilinear(img, out):
    ''' Interpolate the size of the input image using bilinear scheme
    
    This interpolation algorithm does not preserve the SNR unless the image
    is pre-filtered.

    >>> from arachnid.core.image import ndimage_interpolate
    >>> img = numpy.ones((32,32))
    >>> simg = ndimage_interpolate.interpolate_bilinear(img, (15,15))
    >>> print simg.shape
    (15,15)
    
    .. note::
        
        This code is not idential to SPIDER an may give different results
        in some cases.
        
        See: http://www.wadsworth.org/spider_doc/spider/docs/man/ip.html
    
    :Parameters:
    
    img : array
          Image
    out : array or float or list
          Output array or float factor or list of dimensions
          
    :Returns:
    
    out : array
          Interpolated image
    '''
    
    img = numpy.asarray(img, dtype=numpy.float32)
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Only interpolation of 2D and 3D images supported: input has %d-D"%img.ndim
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): shape = (int(out[0]), int(out[1]), int(out[2])) if img.ndim == 3 else (int(out[0]), int(out[1]))
        else: shape = (int(img.shape[0]/out), int(img.shape[1]/out), int(img.shape[2]/out)) if img.ndim == 3 else (int(img.shape[0]/out), int(img.shape[1]/out))
        out = numpy.zeros(shape, dtype=img.dtype)
    else:
        if out.dtype != numpy.float32: raise ValueError, "Requires float32 for out"
    if img.ndim == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
        out = out.reshape((out.shape[0], out.shape[1], 1))
        _spider_interpolate.interpolate_bi_3(img.T, out.T)
        out = out.reshape((out.shape[0], out.shape[1]))
    else:
        _spider_interpolate.interpolate_bi_3(img.T, out.T)
    return out

def _modnx(s): return s+2-numpy.mod(s, 2)

def interpolate_ft(img, out):
    ''' Interpolate the size of the input image using Fourier scheme
    
    This interpolation algorithm is exact in that the total energy is preserved.
    
    >>> from arachnid.core.image import ndimage_interpolate
    >>> img = numpy.ones((32,32))
    >>> simg = ndimage_interpolate.interpolate_ft(img, (15,15))
    >>> print simg.shape
    (15,15)
    
    .. note::
    
        This code gives a slightly different answer than SPIDER, likely due to interpolation
        error.
        
        See: http://www.wadsworth.org/spider_doc/spider/docs/man/ipft.html
    
    :Parameters:
    
    img : array
          Image
    out : array or float or list
          Output array or float factor or list of dimensions
          
    :Returns:
    
    out : array
          Interpolated image
    '''
    
    img = numpy.asarray(img, dtype=numpy.float32)
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Only interpolation of 2D and 3D images supported: input has %d-D"%img.ndim
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): 
            nx = int(out[-1])
            shape = (int(out[0]), int(out[1]), _modnx(int(out[2]))) if img.ndim == 3 else (int(out[0]), _modnx(int(out[1])))
        else: 
            nx = int(img.shape[-1]/out)
            shape = (int(img.shape[0]/out), int(img.shape[1]/out), _modnx(int(img.shape[2]/out))) if img.ndim == 3 else (int(img.shape[0]/out), _modnx(int(img.shape[1]/out)))
        out = numpy.zeros(shape, dtype=img.dtype)
    else:
        if out.dtype != numpy.float32: raise ValueError, "Requires float32 for out"
    if img.ndim == 2:
        img2 = numpy.zeros((img.shape[0], _modnx(img.shape[1])), dtype=numpy.float32)
        img2[:img.shape[0], :img.shape[1]]=img
        _spider_interpolate.finterpolate2(img2.T, out.T, img.shape[1], nx) #img.shape[2], img.shape[1], img.shape[0]
        out = out[:, :nx]
    else:
        img2 = numpy.zeros((img.shape[0], img.shape[1], _modnx(img.shape[2])), dtype=numpy.float32)
        img2[:img.shape[0], :img.shape[1], :img.shape[2]]=img
        _spider_interpolate.finterpolate3(img2.T, out.T, img.shape[2], nx)
        out = out[:shape[0], :shape[1], :nx]
    return out

def interpolate_fs(img, out):
    ''' Interpolate the size of the input image using bicubic/tricubic splines in Fourier space
    
    The boundaries of the new grid coincide with the old grid.
    
    >>> from arachnid.core.image import ndimage_interpolate
    >>> img = numpy.ones((32,32))
    >>> simg = ndimage_interpolate.interpolate_fs(img, (15,15))
    >>> print simg.shape
    (15,15)
    
    .. note::
        
        This code returns the same image as SPIDER.
        
        See: http://www.wadsworth.org/spider_doc/spider/docs/man/ipfs.html
    
    :Parameters:
    
    img : array
          Image
    out : array or float or list
          Output array or float factor or list of dimensions
          
    :Returns:
    
    out : array
          Interpolated image
    '''
    
    img = numpy.asarray(img, dtype=numpy.float32)
    if img.ndim != 3 and img.ndim != 2: raise ValueError, "Only interpolation of 2D and 3D images supported: input has %d-D"%img.ndim
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): 
            shape = (int(out[0]), int(out[1]), int(out[2])) if img.ndim == 3 else (int(out[0]), int(out[1]))
        else: 
            shape = (int(img.shape[0]/out), int(img.shape[1]/out), int(img.shape[2]/out)) if img.ndim == 3 else (int(img.shape[0]/out), int(img.shape[1]/out))
        out = numpy.zeros(shape, dtype=img.dtype)
    else:
        if out.dtype != numpy.float32: raise ValueError, "Requires float32 for out"
    if img.ndim == 2:
        img2 = numpy.zeros((img.shape[0], _modnx(img.shape[1])), dtype=numpy.float32)
        img2[:img.shape[0], :img.shape[1]]=img
        _spider_interpolate.interpolate_fs_2(img2.T, out.T, img.shape[1]) #img.shape[2], img.shape[1], img.shape[0]
    else:
        img2 = numpy.zeros((img.shape[0], img.shape[1], _modnx(img.shape[2])), dtype=numpy.float32)
        img2[:img.shape[0], :img.shape[1], :img.shape[2]]=img
        _spider_interpolate.interpolate_fs_3(img2.T, out.T, img.shape[2])
    return out
