''' Transforms for the SPIDER convention

This module contains utilities perform conversions of rotation and translation information
in the SPIDER convention.

The SPIDER euler angle convention is ZYZ where the angles are denoted as PHI, THETA, PSI, respectively.

Since the rotation is applied in 2D rather than 3D, the PSI angle is usually 0. The 2D alignment parameter convention
is to apply the rotation first, followed by the translation (TR).

.. Created on Jan 22, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy
import healpix
import transforms
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def coarse_angles_3D(resolution, align, half=False, out=None):
    ''' Move project alignment parameters to a coarser grid.
    
    This updates the in-plane rotation and translations to ensure the projections have
    maximum similarity.
    
    :Parameters:
    
        resolution : int
                     Healpix order
        align : array
                2D array where rows are images and columns are
                the following alignment parameters: PSI,THETA,PHI,IN-PLANE,x-translation,y-translation 
                and optionally REF-NUM
        half : bool
               Consider only the half-sphere
        out : array, optional
              Output array for the coarse-grained alignment parameters
    
    :Returns:
        
        out : array
              Coarse-grained alignment parameters:
              PSI,THETA,PHI,IN-PLANE,x-translation,y-translation 
              and optionally REF-NUM (note, PSI is 0)
    '''
    
    _logger.critical("New code")
    ang = healpix.angles(resolution)
    ang[:, 0]=90.0-ang[:, 0]
    resolution = pow(2, resolution)
    if out is None: out=numpy.zeros((len(align), len(align[0])))
    cols = out.shape[1]
    for i in xrange(len(align)):
        theta, phi = healpix.ensure_valid_deg(90.0-align[i,1], align[i,2], half)
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(theta), numpy.deg2rad(phi))
        rot = rotate_into_frame(ang[ipix], (align[i, 0], theta, phi))
        out[i, :3]=(rot, theta, phi)
        if cols>6: out[i, 6] = ipix
    return out

def coarse_angles(resolution, align, half=False, out=None): # The bitterness of men who fear the way of human progress
    ''' Move project alignment parameters to a coarser grid.
    
    This updates the in-plane rotation and translations to ensure the projections have
    maximum similarity.
    
    :Parameters:
    
        resolution : int
                     Healpix order
        align : array
                2D array where rows are images and columns are
                the following alignment parameters: PSI,THETA,PHI,IN-PLANE,x-translation,y-translation 
                and optionally REF-NUM
        half : bool
               Consider only the half-sphere
        out : array, optional
              Output array for the coarse-grained alignment parameters
    
    :Returns:
        
        out : array
              Coarse-grained alignment parameters:
              PSI,THETA,PHI,IN-PLANE,x-translation,y-translation 
              and optionally REF-NUM (note, PSI is 0)
    '''
    
    _logger.critical("New code")
    ang = healpix.angles(resolution)
    ang[:, 0]=90.0-ang[:, 0]
    resolution = pow(2, resolution)
    if out is None: out=numpy.zeros((len(align), len(align[0])))
    cols = out.shape[1]
    for i in xrange(len(align)):
        theta, phi = healpix.ensure_valid_deg(90.0-align[i,1], align[i,2], half)
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(theta), numpy.deg2rad(phi))
        rot, tx, ty = rotate_into_frame_2d(ang[ipix], theta, phi, align[i, 3], align[i,4], align[i,5])
        out[i, 1:6]=( theta, phi, rot, tx, ty)
        if cols>6: out[i, 6] = ipix
    return out

def rotate_into_frame_2d(frame, theta, phi, inplane, dx, dy):
    ''' Fix!
    '''
    
    from ..image import rotate
    rang = rotate.rotate_euler(frame, (-inplane, theta, phi))
    rot = (rang[0]+rang[2])
    rt3d = align_param_2D_to_3D(inplane, dx, dy)
    #return align_param_2D_to_3D(rot, rt3d[1], rt3d[2])
    return align_param_3D_to_2D(rot, rt3d[1], rt3d[2])

def rotate_into_frame(frame, curr):
    '''
    '''
    
    from ..image import rotate
    rang = rotate.rotate_euler(frame, curr)
    return (rang[0]+rang[2])
    
def euler_geodesic_distance(euler1, euler2):
    ''' Calculate the geodesic distance between two unit quaternions
    
    :Parameters:
    
        euler1 : array
                 First Euler set
        euler2 : array
                 Second Euler set
    
    :Returns:
        
        dist : array or float
               Geodesic distance between quaternions
    '''
    
    euler1 = euler1.squeeze()
    euler2 = euler2.squeeze()
    if euler1.ndim == 2 and euler2.ndim==2:
        if euler1.shape[0] != euler2.shape[0]: raise ValueError, "Requires to arrays of the same length"
        dist = numpy.zeros(euler1.shape[0])
        for i in xrange(euler1.shape[0]): dist[i] = euler_geodesic_distance(euler1[i], euler2[i])
        return dist
    euler1 = numpy.deg2rad(euler1)
    euler2 = numpy.deg2rad(euler2)
    q1 = transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
    q2 = transforms.quaternion_from_euler(euler2[0], euler2[1], euler2[2], 'rzyz')
    return numpy.rad2deg(quaternion_geodesic_distance(q1, q2))

def quaternion_geodesic_distance(q1, q2):
    ''' Calculate the geodesic distance between two unit quaternions
    
    :Parameters:
        
        q1 : array
             First quaternion
        q2 : array
             Second quaternion
    
    :Returns:
        
        dist : array or float
               Geodesic distance between quaternions
    '''
    
    q1 = q1.squeeze()
    q2 = q2.squeeze()
    if q1.ndim > 2: raise ValueError, "more than 2 dimensions not supported for q1"
    if q2.ndim > 2: raise ValueError, "more than 2 dimensions not supported for q2"
    if q1.ndim > 1 and q2.ndim > 1: raise ValueError, "Only single list of quaternions supported"
    if q1.ndim == 1: q1, q2 = q2, q1
    if q2.shape[0] != 4: raise ValueError, "q2 does not have 4 elements: %d"%q2.shape[0]
    if q1.ndim == 2:
        if q1.shape[1] != 4: raise ValueError, "q1 does not have 4 elements"
        return 2*numpy.arccos(numpy.dot(q1, q2))
    else:
        if q1.shape[0] != 4: raise ValueError, "q1 does not have 4 elements"
        v = numpy.dot(q1, q2)
        if numpy.allclose(v, 1.0): return 0.0
        return 2*numpy.arccos(v)

def align_param_3D_to_2D(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
    .. note::
        
        RT -> TR
    
    :Parameters:
    
        rot : float
              In plane rotation (RT)
        tx : float
             Translation in the x-direction (RT)
        ty : float
             Translation in the y-direction (RT)
         
    :Returns:
        
        psi : float
              PSI angle (TR)
        sx : float
             Translation in the x-direction (TR)
        sy : float
             Translation in the y-direction (TR)
    '''
    
    rot1 = numpy.deg2rad(rot)
    ca = numpy.cos(rot1)
    sa = numpy.sin(rot1)
    sx = tx*ca - ty*sa
    sy = tx*sa + ty*ca
    return -rot, sx, sy
    
def align_param_2D_to_3D(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
    .. note::
        
        TR -> RT
    
    :Parameters:
        
        rot : float
              In plane rotation (TR)
        tx : float
             Translation in the x-direction (TR)
        ty : float
             Translation in the y-direction (TR)
         
    :Returns:
        
        psi : float
              PSI angle (RT)
        sx : float
             Translation in the x-direction (RT)
        sy : float
             Translation in the y-direction (RT)
    '''
    
    rot1 = -numpy.deg2rad(rot)
    ca = numpy.cos(rot1)
    sa = numpy.sin(rot1)
    sx =  tx*ca + ty*sa
    sy = -tx*sa + ty*ca 
    return -rot, sx, sy

def euler_to_vector(phi, theta):
    ''' Convert Euler angles to a vector
    
    .. note::
        
        http://xmipp.cnb.csic.es/~xmipp/trunk/xmipp/documentation/html/geometry_8cpp_source.html
        
    :Parameters:
    
    phi : float
          Phi angle in degrees (psi,theta,phi = ZYZ) 
    theta : float
            Theta angle in degrees (psi,theta,phi = ZYZ)
    
    :Returns:
    
    sc : float
         Something
    ss : float
         Something
    ct : float
         Something
    '''
    
    theta = numpy.deg2rad(theta)
    phi = numpy.deg2rad(phi)
    cp = numpy.cos(phi)   # ca
    ct = numpy.cos(theta) # cb
    sp = numpy.sin(phi)   # sa
    st = numpy.sin(theta) # sb
    sc = st * cp          #sb * ca;
    ss = st * sp          #sb * sa;
    return sc, ss, ct


