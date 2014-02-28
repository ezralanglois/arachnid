''' pyHEALPix Library

It works with Euler angles in the YZ rotating frame convention where (THETA,PHI). The
angles must be in radians and in the following range:

- 0 <= THETA < PI (also called colatitude)
- 0 <= PHI < 2PI  (also called longitude)

.. note::
    
    The representation is compatiable with the SPIDER ZYZ rotating rame (PSI,THETA,PHI).

The following table relates the resolution parameter (and its nside counterpart) to the sampling
statistics:

==========     =====       =====    =====      =====================  ===========  ===============
Resolution     nside       total    theta      half sphere (equator)  half sphere  half sphere sum
----------     -----       -----    -----      ---------------------  -----------  ---------------
1              2           48       29.32      28                     20           20
2              4           192      14.66      104                    88           108
3              8           768      7.33       400                    368          476
4              16          3072     3.66       1568                   1504         1980
5              32          12288    1.83       6208                   6080         8060
6              64          49152    0.92       24704                  24448        32508
7              128         196608   0.46       98560                  98048        130556
8              256         786432   0.23       393728                 392704       523260
==========     =====       =====    =====      =====================  ===========  ===============

.. todo::
    
    1. use pmod to ensure angles in proper range

.. Created on Aug 17, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
import numpy
import math


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try:
    from core import _healpix
    _healpix;
except:
    #_logger.addHandler(logging.StreamHandler())
    #_logger.exception("failed to import _healpix")
    from ..app import tracing
    tracing.log_import_error("Failed to import pyHEALPix module - certain functionality will not be available", _logger)

def angles(resolution, half=False, out=None):
    ''' Generate a list of angles on SO(2) using HEALPix
    
    Assumes Theta and PHI are in the ZYZ rotating frame where
    (PSI,THETA,PHI), respectively.
    
    >>> from arachnid.core.orient.healpix import *
    >>> angles(1)
    [[   0.           23.55646431   45.        ]
     [   0.           23.55646431  135.        ]
     ...
     
    >>> len(angles(1))
    48
    >>> len(angles(1, True))
    20
     
    >>> angles(1, out=numpy.zeros((1,3))
    [[   0.           23.55646431   45.        ]]
    
    :Parameters:
    
        resolution : int
                     Healpix resolution or sampling rate on the sphere
        half : bool
               Sample only from the half sphere (not equater projections)
        out : array, optional
              A nx3 array of angles on SO(2) (psi,theta,phi), where psi
              zero
    
    :Returns:
        
        out : array
              A nx3 array of angles on SO(2) (psi,theta,phi), where psi
              zero
    '''
    
    nsample = pow(2, resolution)
    npix = 12*nsample*nsample if not half else 6*nsample*nsample - nsample*2 # +nsample*2 add the equator projections
    ang = numpy.zeros(2)
    if out is None:
        out = numpy.zeros((npix, 3))
    for i in xrange(out.shape[0]):
        _healpix.pix2ang_ring(nsample, i, ang)
        out[i, 1:]=numpy.rad2deg(ang)
    return out
    
def angles_gen(resolution, deg=False, half=False):
    ''' Generator that lists angles on SO(2) using HEALPix
    
    Assumes Theta and PHI are in the ZYZ rotating frame where
    (PSI,THETA,PHI), respectively.
    
    >>> from arachnid.core.orient.healpix import *
    >>> [a for a in angles_gen(1)]
    [[   0.           23.55646431   45.        ]
     [   0.           23.55646431  135.        ]
     ...
    
    :Parameters:
    
        resolution : int
                     Sampling resolution 
        deg : bool
              Convert radians to degrees
        half : bool
               From half sphere (no equator)
    
    :Returns:
        
        array : array
                A 1x3 array of angles on SO(2) (psi,theta,phi), where psi
                zero
    '''
    
    nsample = pow(2, resolution)
    npix = 12*nsample*nsample if not half else 6*nsample*nsample - nsample*2
    ang = numpy.zeros(2)
    for i in xrange(npix):
        ang = _healpix.pix2ang_ring(nsample, i, ang)
        yield numpy.rad2deg(ang) if deg else ang

def res2npix(resolution, half=False, equator=False):
    ''' Get the number of pixels for a given resolution
    
    >>> from arachnid.core.orient.healpix import *
    >>> res2npix(1)
    48
    >>> res2npix(1, True)
    20
    >>> res2npix(1, True, True)
    28
    
    :Parameters:
        
        resolution : int
                     Sampling resolution 
        half : bool
               From half sphere
        equator : bool
                 Include equator pixels
    
    :Returns:
        
        npix : int
               Number of pixels for a given resolution
    '''
    
    nsample = pow(2, resolution)
    if half:
        return 6*nsample*nsample + nsample*2 if equator else 6*nsample*nsample - nsample*2
    return 12*nsample*nsample

def theta2nside(theta, max_res=8):
    '''  Given a theta increment, find the closest
    healpix sampling index.
    
    :Parameters:
    
        theta : float
                Sampling on theta
        max_res : float
                  Max resolution to return
    
    :Returns:
    
        resolution : int
                     Resolution for theta sampling
    '''
    
    area = numpy.zeros(max_res)
    for i in xrange(1, area.shape[0]):
        area[i] = nside2pixarea(i)
    return numpy.argmin(numpy.abs(theta-area))+1

def pmod(x, y):
    ''' Modules result that is always positive
    
    :Parameters:
    
        x : float
            Number
        y : float
            Number
    
    :Returns:
        
        out : float
              Positive modulus of x%y
    '''
    
    if y == 0: return x
    return x - y * math.floor(float(x)/y)
    
def nside2pixarea(resolution, degrees=False):
    """Give pixel area given nside.

    .. note::
        
        Raise a ValueError exception if nside is not valid.

    Examples:
    
    >>> from arachnid.core.orient.healpix import *
    >>> nside2pixarea(128, degrees = True)
    0.2098234113027917

    >>> nside2pixarea(256)
    1.5978966540475428e-05

    :Parameters:
    
        resolution : int
                     nside = 2**resolution
        degrees : bool
                  if True, returns pixel area in square degrees, in square radians otherwise

    :Returns:
    
        pixarea : float
                  pixel area in suqare radian or square degree
    """
    
    nsample = pow(2, resolution)
    npix = 12*nsample*nsample
    pixarea = 4*numpy.pi/npix
    if degrees: pixarea = numpy.rad2deg(numpy.rad2deg(pixarea))
    return numpy.sqrt(pixarea)

def pix2ang(resolution, pix, scheme='ring', half=False, out=None):
    ''' Convert Euler angles to pixel
    
    :Parameters:
        
        resolution : int
                     Pixel resolution
        theta : float or array
                Euler angle theta or array of Euler angles
        phi : float
              Euler angle phi or optional if theta is array of both
        scheme : str
                 Pixel layout scheme: nest or ring
        half : bool
               Convert Euler angles to half volume
        out : array, optional
              Array of pixels for specified array of Euler angles
    
    :Returns:
        
        out : in or, array
              Pixel for specified Euler angles
    '''
    
    resolution = pow(2, resolution)
    if scheme not in ('nest', 'ring'): raise ValueError, "scheme must be nest or ring"
    _pix2ang = getattr(_healpix, 'pix2ang_%s'%scheme)
    if hasattr(pix, '__iter__'):
        if out is None: out = numpy.zeros((len(pix), 2))
        for i in xrange(len(pix)):
            out[i, :] = _pix2ang(int(resolution), int(pix[i]))
        return out
    else:
        return _pix2ang(int(resolution), int(pix))
    
def healpix_euler_rad(ang):
    ''' Ensure Euler angles in radians fall in 
    the accepted healpix range.
    
    .. note::
        
        This currently does not handel theta > 3/4PI or theta < 0
    
    :Parameters:
    
        ang : tuple
              Theta, PI in radians
    
    :Returns:
    
        theta : float
                Theta between 0 and PI in radians
        phi : float
                PHI between 0 and 2PI in radians
    '''
    
    if len(ang) == 2:
        theta, phi = ang
        twopi = numpy.pi*2
        assert(theta>=0)
        if phi < 0: phi += twopi
        if theta > numpy.pi:
            theta = theta-numpy.pi/2
            phi += numpy.pi
            if phi > twopi: phi-=twopi
        return theta, phi
    else: raise ValueError, "Not implemented for other than 2 angles"
    
def healpix_euler_deg(ang):
    ''' Ensure Euler angles in degrees fall in 
    the accepted healpix range.
    
    .. note::
        
        This currently does not handel theta > 270.0 or theta < 0
    
    :Parameters:
    
        ang : tuple
              Theta, PI in degrees
    
    :Returns:
    
        theta : float
                Theta between 0 and 180 in degrees
        phi : float
                PHI between 0 and 360 in degrees
    '''
    
    if len(ang) == 2:
        theta, phi = ang
        assert(theta>=0)
        if phi < 0: phi += 360.0
        if theta > 180.0:
            theta = theta-90.0
            phi += 180.0
            if phi > 360.0: phi-=360.0
        return theta, phi
    else: raise ValueError, "Not implemented for other than 2 angles"
    
def healpix_half_sphere_euler_rad(ang):
    ''' Ensure Euler angles in radians fall in 
    the accepted healpix range on the half sphere.
    
    .. note::
        
        This currently does not handel theta > 3/4PI or theta < 0
    
    :Parameters:
    
        ang : tuple
              Theta, PI in radians
    
    :Returns:
    
        theta : float
                Theta between 0 and 90 in radians
        phi : float
                PHI between 0 and 360 in radians
    '''
    
    if len(ang) == 2:
        halfpi = numpy.pi/2
        twopi = numpy.pi*2
        theta, phi = ang
        assert(theta>=0)
        if phi < 0: phi += twopi
        if theta <= numpy.pi and theta > halfpi:
            theta = twopi-theta
            phi += numpy.pi
            if phi > twopi: phi -= twopi
        if theta > numpy.pi:
            theta -= numpy.pi
        return theta, phi
    else: raise ValueError, "Not implemented for other than 2 angles"

def ang2pix(resolution, theta, phi=None, scheme='ring', half=False, deg=False, out=None):
    ''' Convert Euler angles to pixel
    
    :Parameters:
    
        resolution : int
                     Pixel resolution
        theta : float or array
                Euler angle theta or array of Euler angles (colatitude)
        phi : float
              Euler angle phi or optional if theta is array of both (longitude)
        scheme : str
                 Pixel layout scheme: nest or ring
        half : bool
               Convert Euler angles to half volume
        deg : bool
              Angles in degrees
        out : array, optional
              Array of pixels for specified array of Euler angles
    
    :Returns:
        
        out : in or, array
              Pixel for specified Euler angles
    '''
    
    twopi=numpy.pi*2
    resolution = pow(2, resolution)
    if scheme not in ('nest', 'ring'): raise ValueError, "scheme must be nest or ring"
    if hasattr(theta, '__iter__'):
        if phi is not None and not hasattr(phi, '__iter__'): 
            raise ValueError, "phi must be None or array when theta is an array"
        if hasattr(phi, '__iter__'): theta = zip(theta, phi)
        _ang2pix = getattr(_healpix, 'ang2pix_%s'%scheme)
        if out is None: out = numpy.zeros(len(theta), dtype=numpy.long)
        i = 0
        for t, p in theta:
            if deg: t, p = numpy.deg2rad((t, p))
            if half:
                t, p = healpix_half_sphere_euler_rad((t, p))
            else:
                t, p = healpix_euler_rad((t, p))
            if t > numpy.pi: raise ValueError, "Invalid theta: %f, must be less than PI"%t
            if t < 0: raise ValueError, "Invalid theta: %f, must be greater than 0"%t
            if p > twopi: raise ValueError, "Invalid phi: %f, must be less than PI"%p
            if p < 0: raise ValueError, "Invalid phi: %f, must be greater than 0"%p
            out[i] = _ang2pix(int(resolution), float(t), float(p))
            i += 1
        return out
    else:
        _ang2pix = getattr(_healpix, 'ang2pix_%s'%scheme)
        if phi is None: "phi must not be None when theta is a float"
        if deg: theta, phi = numpy.deg2rad((theta, phi))
        if half:
            theta, phi = healpix_half_sphere_euler_rad((theta, phi))
        else:
            theta, phi = healpix_euler_rad((theta, phi))
        if theta > numpy.pi: raise ValueError, "Invalid theta: %f, must be less than PI"%theta
        if theta < 0: raise ValueError, "Invalid theta: %f, must be greater than 0"%theta
        if phi > twopi: raise ValueError, "Invalid phi: %f, must be less than PI"%phi
        if phi < 0: raise ValueError, "Invalid phi: %f, must be greater than 0"%phi
        return _ang2pix(int(resolution), float(theta), float(phi))




