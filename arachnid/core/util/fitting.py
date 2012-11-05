''' Utilities to fit a curve to a set of data

.. Created on Oct 5, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, scipy.optimize
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def fit_linear_interp(fsc_curve, fsc_value):
    ''' Determine the spatial frequency raw from the FSC curve
    
    :Parameters:
    
    fsc_curve : array
                FSC curve
    fsc_value : float
                FSC value to choose spatial frequency
    
    :Returns:
    
    sp : float
         Spatial frequency at fsc_value
    '''
    
    if isinstance(fsc_curve, tuple):
        x, y = fsc_curve
    else:
        x, y = fsc_curve[:, 0], fsc_curve[:, 1]
    
    idx = numpy.argsort(y)
    ridx = idx[numpy.searchsorted(y[idx], fsc_value, 'left')]
    if y[ridx] > 0.5 and (ridx+1) < len(y): lidx = ridx+1
    elif ridx > 0: lidx = ridx-1
    else: lidx=ridx
    try:
        func = numpy.poly1d(numpy.polyfit([y[ridx], y[lidx]], [x[ridx], x[lidx]], 1))
    except:
        _logger.error("%d > %d --- %d > %d"%(ridx, len(y), lidx, len(y)))
        raise
    return func(fsc_value)

def fit_sigmoid_interp(fsc_curve, fsc_value):
    ''' Determine the spatial frequency raw from the FSC curve (parametric sigmoid model)
    
    :Parameters:
    
    fsc_curve : array
                FSC curve
    fsc_value : float
                FSC value to choose spatial frequency
    
    :Returns:
    
    sp : float
         Spatial frequency at fsc_value
    '''
    
    coeff = fit_sigmoid(fsc_curve[:, 0], fsc_curve[:, 1])
    return sigmoid(coeff, fsc_value)

def fit_sigmoid(x, y):
    ''' Use non-linear least squares to fit x and y to a sigmoid-like function
    
    :Parameters:
    
    x : array
        X-values for training
    y : array
        y-values for training
    
    :Returns:
    
    coeff : array
            Array of coefficents that fit x to y
    '''
    
    def errfunc(p,x,y): return y-sigmoid(p,x)
    p0 = [0.5, 0.5, 0.5, 0.5]
    return scipy.optimize.leastsq(errfunc,p0,args=(x, y))[0]

def sigmoid_inv(coeff,y):
    ''' Returns a the related inverse of the sigmoid-like function for the given coeffients and x values
    
    :Parameters:
    
    coeff : array
            Array of coeffients
    y : array
        Array of y-values
    
    :Returns:
    
    x : array
        Inverse of sigmoid like function of y-values
    '''
    
    return numpy.log( coeff[1]/(coeff[2]-y) - 1.0 )/-coeff[0] - coeff[3]

def sigmoid(coeff,x):
    ''' Returns a sigmoid-like function for the given coeffients and x values
    
    :Parameters:
    
    coeff : array
            Array of coeffients
    x : array
        Array of x-values
    
    :Returns:
    
    y : array
        Sigmoid like function of x values
    '''
    
    return coeff[2] - ( coeff[1] / (1.0 + numpy.exp(-coeff[0]*(x+coeff[3])) ))


