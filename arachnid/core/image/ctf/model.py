''' Construct a model of the CTF

.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy
from .. import ndimage_utility

def _transfer_function(img, defocus, alpha, cs, lam, f0, bfactor=0.0):
    ''' Contrast transfer function
    
    :Parameters:
        
        img : array
              Radial index squared
        defocus : float
                  Defocus in angstroms
        alpha : float
                Amplitude contrast in radians
        cs : float
             Spherical abberation in A
        lam : float
              Electron energy in A
        f0 : float
                Inverse of pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        out : array
              Transfer Function
    '''
    
    k2=-defocus*numpy.pi*lam*f0**2
    if cs == 0.0:
        h = numpy.sin(k2*img-alpha)
    else:
        k4 = 0.5*numpy.pi*lam**3*cs*f0**4
        h = numpy.sin(k2*img+k4*img*img-alpha)
    if bfactor != 0.0:
        kr=f0**2*bfactor
        h *= numpy.exp(-kr*img)
    return h

def transfer_function(img, defocus, ampcont, cs, voltage, apix, bfactor=0.0):
    ''' Contrast transfer function
    
    :Parameters:
        
        img : array
              Radial index squared
        defocus : float
                  Defocus in angstroms
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        out : array
              Transfer Function
    '''
    
    cs *= 1e7
    f0 = 1.0/apix
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    alpha = numpy.arctan((ampcont/(1.0-ampcont)))
    return _transfer_function(img, defocus, alpha, cs, lam, f0, bfactor)

def transfer_function_1D(n, defocus, ampcont, cs, voltage, apix, bfactor=0.0):
    ''' Contrast transfer function in 1D
    
    :Parameters:
        
        n : int
            Size of the contrast transfer function
        defocus : float
                  Defocus in angstroms
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        out : array
              Transfer Function
    '''
    
    out = (numpy.arange(n, dtype=numpy.float)/float(n)/2.0)**2
    out[:] = transfer_function(out, defocus, ampcont, cs, voltage, apix, bfactor)
    return out

def transfer_function_2D(n, defocus, ampcont, cs, voltage, apix, bfactor=0.0, **extra):
    ''' Contrast transfer function in 2D - no astigmatism
    
    :Parameters:
        
        n : tuple or int
            Size of the contrast transfer function
        defocus : float
                  Defocus in angstroms
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        out : array
              Transfer Function
    '''
    
    if not isinstance(n, tuple): n = (n, n)
    out = ndimage_utility.radial_image_fft(n, norm=True)
    out[:] = transfer_function(out, defocus, ampcont, cs, voltage, apix, bfactor)
    return out

def transfer_function_2D_full(n, defocus_u, defocus_v, defocus_ang, ampcont, cs, voltage, apix, bfactor=0.0, **extra):
    ''' Contrast transfer function in 2D
    
    :Parameters:
        
        n : tuple or int
            Size of the contrast transfer function
        defocus_u : float
                    Defocus on minor axis in angstroms
        defocus_v : float
                    Defocus on major axis in angstroms
        defocus_ang : float
                      Astigmatism angle in degrees between x-axis and minor defocus axis
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        out : array
              Transfer Function
    '''
    
    if not isinstance(n, tuple): n = (n, n)
    out = ndimage_utility.radial_image_fft(n, norm=True)
    ang = numpy.cos(2.0*(numpy.deg2rad(defocus_ang)-ndimage_utility.angular_image(n)))
    defocus = 0.5*((defocus_u+defocus_v) + ang*(defocus_u-defocus_v))
    out[:] = transfer_function(out, defocus, ampcont, cs, voltage, apix, bfactor)
    return out

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
