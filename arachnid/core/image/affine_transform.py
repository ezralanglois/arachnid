''' Affine transforms for an image

.. Created on Jan 14, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging, numpy


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from spi import _spider_rotate
    _spider_rotate;
except:
    from ..app import tracing
    _spider_rotate=None
    tracing.log_import_error('Failed to load _spider_rotate.so module', _logger)

def rotate_translate_image(img, ang, tx=0.0, ty=0.0, out=None, scale=1.0):
    '''
    '''
    
    if hasattr(ang, '__iter__'):
        tx = ang[6]
        ty = ang[7]
        ang = ang[5]
    
    if out is None: out = img.copy()
    _spider_rotate.rotate_image(img.T, out.T, ang, scale, tx, ty)
    return out

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

