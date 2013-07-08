'''
.. Created on Jun 29, 2013
.. codeauthor:: robertlanglois
'''

import logging
import numpy #, scipy
#import scipy.ndimage.filters

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def gaussian_lowpass(low_cutoff):
    '''
    '''
    
    pass

def gaussian_highpass(high_cutoff):
    '''
    '''
    
    pass

def gaussian_bandpass(low_cutoff, high_cutoff):
    '''
    '''
    
    pass

def butterworth_lowpass(shape, low_cutoff, falloff):
    ''' Generate a butterworth lowpass filter
    
    .. note:: 
        
        Taken from:
        http://www.psychopy.org/epydoc/psychopy.filters-pysrc.html
        and
        SPIDER
    
    :Parameters:
    
    shape : tuple
            Size of the kernel
    low_cutoff : float
                 Low frequency cutoff
    falloff : float
              High frequency falloff
    
    :Returns:
    
    out : array
          Butterworth lowpass kernel
    '''
    
    eps = 0.882
    aa = 10.624
    n = 2 * numpy.log10(eps/numpy.sqrt(aa*aa-1.0))/numpy.log10(low_cutoff/falloff)
    cutoff = low_cutoff/numpy.power(eps, 2.0/n)
    x =  numpy.linspace(-0.5, 0.5, shape[1]) 
    y =  numpy.linspace(-0.5, 0.5, shape[0])
    radius = numpy.sqrt((x**2)[numpy.newaxis] + (y**2)[:, numpy.newaxis])
    return numpy.sqrt( 1.0 / (1.0 + (radius / cutoff)**n) )

def butterworth_highpass(high_cutoff, falloff):
    '''
    '''
    
    pass

def butterworth_bandpass(low_cutoff, high_cutoff, falloff):
    '''
    '''
    
    pass

"""
def raised_cosine_lowpass(sigma_low):
    '''
    '''
    
    pass

def raised_cosine_highpass(sigma_high):
    '''
    '''
    
    pass

def raised_cosine_bandpass(sigma_low, sigma_high):
    '''
    '''
    
    pass
"""

