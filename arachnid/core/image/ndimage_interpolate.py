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
import sys

_ndinter = sys.modules[__name__]

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
    
def sincblackman(bin_factor, template_min = 15, kernel_size=2002, dtype=numpy.float32):
    '''
    '''
    
    assert(bin_factor > 1.0)
    bin_factor = 1.0/bin_factor
    frequency_cutoff = 0.5*bin_factor
    kernel = numpy.zeros(kernel_size, dtype=dtype)
    _resample.sinc_blackman_kernel(kernel, int(template_min), float(frequency_cutoff))
    return kernel

def downsample(img, out, kernel=None):
    '''
    '''
    
    if numpy.dtype(img.dtype).kind != 'f': raise ValueError, "Input array must be floating point, not %s"%numpy.dtype(img.dtype).kind
    scale=None
    bin_factor=None
    if not hasattr(out, 'ndim'):
        if hasattr(out, '__len__'): shape = (int(out[0]), int(out[1]), int(out[2])) if img.ndim == 3 else (int(out[0]), int(out[1]))
        else:
            assert(out > 1.0)
            scale = 1.0/out 
            bin_factor=out
            shape = (int(img.shape[0]/out), int(img.shape[1]/out), int(img.shape[2]/out)) if img.ndim == 3 else (int(img.shape[0]/out), int(img.shape[1]/out))
        out = numpy.zeros(shape, dtype=img.dtype)
    else:
        if out.dtype != img.dtype: raise ValueError, "Requires output and input of the same dtype"
    if scale is None: scale = float(out.shape[0])/float(img.shape[0])
    if bin_factor is None: bin_factor = float(img.shape[0])/float(out.shape[0])
    if kernel is None: kernel=sincblackman(bin_factor, dtype=img.dtype)
    if kernel.dtype != img.dtype: raise ValueError, "Requires kernel and input of the same dtype"
    # todo automatic convert to fortran with .T
    _resample.downsample(img, out, kernel, scale)
    return out

interpolate_sblack=downsample
    
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
    
    if method not in ('bilinear', 'ft', 'fs', 'sblack'):
        raise ValueError, "method argument must be one of the following: bilinear,ft,fs,sblack"
    return getattr(_ndinter, 'interpolate_'+method)(img, out)

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
