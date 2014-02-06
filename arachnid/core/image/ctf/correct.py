'''
.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import scipy.fftpack
import numpy
import logging
import math
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from ..spi import _spider_ctf
    _spider_ctf;
except:
    _spider_ctf=None
    from arachnid.core.app import tracing
    tracing.log_import_error('Failed to load _spider_ctf.so module', _logger)


def spider_fft2(img):
    '''
    '''
    
    nsam = (img.shape[1]+2) if (img.shape[1]%2) == 0 else (img.shape[1]+1)
    out = numpy.zeros((img.shape[0], nsam), dtype=img.dtype)
    out[:, :img.shape[1]] = img
    _spider_ctf.fft2_image(out.T, img.shape[0])
    return out

def correct_model(img, ctfimg, fourier=False):
    '''
    '''
    
    img = scipy.fftpack.fftshift(scipy.fftpack.fft2(img))
    if ctfimg is not None: img *= ctfimg
    if fourier: return img
    return scipy.fftpack.ifft2(scipy.fftpack.ifftshift(img))
    
def correct(img, ctfimg, fourier=False):
    ''' Corret the CTF of an image
    
    :Parameters:
    
    :Returns:
    
    '''
    
    #ctfimg = numpy.require(ctfimg, dtype=numpy.float64)
    #img = numpy.require(img, dtype=numpy.float64)
    nsam = (img.shape[1]+2) if (img.shape[1]%2) == 0 else (img.shape[1]+1)
    out = numpy.zeros((img.shape[0], nsam), dtype=img.dtype)
    out[:, :img.shape[1]] = img
    if fourier:
        if ctfimg is None:
            _spider_ctf.fft2_image(out.T, img.shape[0])
            out = out.ravel()[::2] + 1j*out.ravel()[1::2]
            return out.ravel()[:img.shape[0]/2*nsam].reshape((img.shape[0]/2, nsam))
        
        try:
            _spider_ctf.correct_image_fourier(out.T, ctfimg.T, img.shape[0])
        except:
            _logger.error("out dtype: %s"%str(out.dtype))
            _logger.error("ctfimg dtype: %s"%str(ctfimg.dtype))
            raise
        out = out.ravel()[::2] + 1j*out.ravel()[1::2]
        return out.ravel()[:img.shape[0]/2*nsam].reshape((img.shape[0]/2, nsam))
    else:
        try:
            _spider_ctf.correct_image(out.T, ctfimg.T, img.shape[0])
        except:
            _logger.error("out dtype: %s"%str(out.dtype))
            _logger.error("ctfimg dtype: %s"%str(ctfimg.dtype))
            raise
        return out[:, :img.shape[1]]

def phase_flip_transfer_function(out, defocus, cs, ampcont, envelope_half_width=10000, voltage=None, elambda=None, apix=None, maximum_spatial_freq=None, source=0.0, defocus_spread=0.0, astigmatism=0.0, azimuth=0.0, ctf_sign=-1.0, **extra):
    ''' Create a transfer function for phase flipping
    
    :Parameters:
    
    out : size, tuple, array
          Size of square ransfer function image, tuple of dimensions, or image
    defocus : float
              Amount of defocus, in Angstroems
    cs : object
         Spherical aberration constant
    ampcont : float
              Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
    envelope_half_width : float
                          Envelope parameter specifies the 2 sigma level of the Gaussian
    voltage : float
              Voltage of microscope (Defaut: None)
    elambda : float
              Wavelength of the electrons (Defaut: None)
    apix : float
           Size of pixel in angstroms  (Defaut: None)
    maximum_spatial_freq : float
                           Spatial frequency radius corresponding to the maximum radius (Defaut: None)
    source : float
             Size of the illumination source in reciprocal Angstroems
    defocus_spread : float
                     Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
    astigmatism : float
                  Defocus difference due to axial astigmatism (Defaut: 0)
    azimuth : float
              Angle, in degrees, that characterizes the direction of astigmatism (Defaut: 0)
    ctf_sign : float
               Application of the transfer function results in contrast reversal if underfocus (Defaut: -1)
    
    :Returns:
    
    out : array
          Transfer function image
    '''
    
    if elambda is None: 
        if voltage is None: raise ValueError, "Wavelength of the electrons is not set as elambda or voltage"
        elambda = 12.398 / math.sqrt(voltage * (1022.0 + voltage))
    if maximum_spatial_freq is None: 
        if apix is None: raise ValueError, "Patial frequency radius corresponding to the maximum radius is not set as maximum_spatial_freq or apix"
        maximum_spatial_freq = 0.5/apix
        
    if isinstance(out, tuple):
        nsam = (out[0]+2)/2 if (out[0]%2) == 0 else (out[0]+1)/2
        out = numpy.zeros((out[1], nsam), dtype=numpy.complex64)
    elif isinstance(out, int):
        nsam = (out+2)/2 if (out%2) == 0 else (out+1)/2
        out = numpy.zeros((out, nsam), dtype=numpy.complex64)
    else:
        if out.dtype != numpy.complex64: raise ValueError, "Requires complex64 for out"
    _spider_ctf.transfer_function_phase_flip_2d(out.T, out.shape[0], float(cs), float(defocus), float(maximum_spatial_freq), float(elambda), float(source), float(defocus_spread), float(astigmatism), float(azimuth), float(ampcont), 0.0, int(ctf_sign))
    return out

def transfer_function(out, defocus, cs, ampcont, envelope_half_width=10000, voltage=None, elambda=None, apix=None, maximum_spatial_freq=None, source=0.0, defocus_spread=0.0, astigmatism=0.0, azimuth=0.0, ctf_sign=-1.0, **extra):
    ''' Create a transfer function for phase flipping
    
    :Parameters:
    
    out : size, tuple, array
          Size of square ransfer function image, tuple of dimensions, or image
    defocus : float
              Amount of defocus, in Angstroems
    cs : object
         Spherical aberration constant
    ampcont : float
              Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
    envelope_half_width : float
                          Envelope parameter specifies the 2 sigma level of the Gaussian
    voltage : float
              Voltage of microscope (Defaut: None)
    elambda : float
              Wavelength of the electrons (Defaut: None)
    apix : float
           Size of pixel in angstroms  (Defaut: None)
    maximum_spatial_freq : float
                           Spatial frequency radius corresponding to the maximum radius (Defaut: None)
    source : float
             Size of the illumination source in reciprocal Angstroems
    defocus_spread : float
                     Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
    astigmatism : float
                  Defocus difference due to axial astigmatism (Defaut: 0)
    azimuth : float
              Angle, in degrees, that characterizes the direction of astigmatism (Defaut: 0)
    ctf_sign : float
               Application of the transfer function results in contrast reversal if underfocus (Defaut: -1)
    
    :Returns:
    
    out : array
          Transfer function image
    '''
    
    if elambda is None: 
        if voltage is None: raise ValueError, "Wavelength of the electrons is not set as elambda or voltage"
        elambda = 12.398 / math.sqrt(voltage * (1022.0 + voltage))
    if maximum_spatial_freq is None: 
        if apix is None: raise ValueError, "Patial frequency radius corresponding to the maximum radius is not set as maximum_spatial_freq or apix"
        maximum_spatial_freq = 0.5/apix
        
    if isinstance(out, tuple):
        nsam = (out[0]+2)/2 if (out[0]%2) == 0 else (out[0]+1)/2
        out = numpy.zeros((out[1], nsam), dtype=numpy.complex64)
    elif isinstance(out, int):
        nsam = (out+2)/2 if (out%2) == 0 else (out+1)/2
        out = numpy.zeros((out, nsam), dtype=numpy.complex64)
    else:
        if out.dtype != numpy.complex64: raise ValueError, "Requires complex64 for out"
    env = envelope_half_width #1./envelope_half_width**2
    _spider_ctf.transfer_function_phase_flip_2d(out.T, out.shape[0], float(cs), float(defocus), float(maximum_spatial_freq), float(elambda), float(source), float(defocus_spread), float(astigmatism), float(azimuth), float(ampcont), env, int(ctf_sign))
    return out



