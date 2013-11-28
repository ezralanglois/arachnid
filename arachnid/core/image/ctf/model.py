''' Construct a model of the CTF

.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy
from .. import ndimage_utility

def model(defocus, sfreq, n, ampcont, cs, voltage, apix):
    ''' CTF model with non-generalized variables dependent on voltage
    
    Useful for CS-correct microscopes when the CS is 0
    
    :Parameters:
    
    defocus : float
              Defocus in microns
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Number of points to sample
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
             
    :Returns:

    ctf : array
          CTF model
    '''
    
    sfreq = sfreq.astype(numpy.float)
    cs *= 1e7
    defocus *= 1e4
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    w = numpy.arcsin(ampcont)
    
    km1 = 1.0 / (2.0*apix)
    dk = km1/n
    sfreq = sfreq*dk
    
    sfreq4 = lam**3*cs*0.5*numpy.pi*sfreq**4
    sfreq2 = lam*numpy.pi*sfreq**2
    
    return numpy.sin( sfreq4 - defocus*sfreq2 + w) #*1e4 + w )

def model_1d(defocus, n, ampcont, cs, voltage, apix):
    ''' CTF model with non-generalized variables dependent on voltage
    
    :Parameters:
    
    defocus : float
              Defocus in microns
    n : int
        Number of points to sample
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
           
    :Returns:

    ctf : array
          1D CTF model
    '''
    
    return model(defocus, numpy.arange(0, n), n/2, ampcont, cs, voltage, apix)

def model_2d(defocus, n, ampcont, cs, voltage, apix):
    ''' CTF model with non-generalized variables dependent on voltage
    
    :Parameters:
    
    defocus : float
              Defocus in microns
    n : int
        Number of points to sample
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
           
    :Returns:

    ctf : array
          2D CTF model
    '''
    
    return model(defocus, numpy.sqrt(ndimage_utility.radial_image((n, n))), n/2, ampcont, cs, voltage, apix)

def bfactor_model(p, freq):
    '''Expoential model describing the background factor decay
    
    :Parameters:
    
    p : array
        Array of parameters
    freq : array
          Array of frequencey values
    
    :Returns:
    
    model : array
            Estimated model
    '''
    
    return p[0] + p[1] * numpy.sqrt(freq) + p[2]*freq + p[3]*numpy.square(freq)
