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

def mirror_x(ang):
    ''' Mirror Euler angles in ZYZ over the x-axis
    
    :Parameters:
    
        ang : tuple
              Either PSI, THETA, PHI or THETA, PHI
    
    :Returns:
    
        out : tuple
              Depends on input: mirrored PSI, THETA, PHI 
              or THETA PHI
    '''
    
    if len(ang) == 2: return (ang[0]-180.0, ang[1])
    else: return (180.0-ang[0], ang[1]-180.0, ang[2])

def mirror_y(ang):
    ''' Mirror Euler angles in ZYZ over the y-axis
    
    :Parameters:
    
        ang : tuple
              Either PSI, THETA, PHI or THETA, PHI
    
    :Returns:
    
        out : tuple
              Depends on input: mirrored PSI, THETA, PHI 
              or THETA PHI
    '''
    
    if len(ang) == 2: return (ang[0]+180.0, ang[1])
    else: return (-ang[0], ang[1]+180.0, ang[2])

def unmirror_y(ang):
    ''' Mirror Euler angles in ZYZ over the y-axis
    
    :Parameters:
    
        ang : tuple
              Either PSI, THETA, PHI or THETA, PHI
    
    :Returns:
    
        out : tuple
              Depends on input: mirrored PSI, THETA, PHI 
              or THETA PHI
    '''
    
    if len(ang) == 2:
        if ang[0] > 180.0:
            return (ang[0]-180.0, ang[1])
        assert(ang[0] >= 0.0)
        return (ang[0], ang[1])
    else: 
        if ang[1] > 180.0:
            return (-ang[0], ang[1]-180.0, ang[2])
        assert(ang[1] >= 0.0)
    return ang
    

def mirror_xy(ang):
    ''' Mirror Euler angles in ZYZ over the y-axis
    
    :Parameters:
    
        ang : tuple
              Either PSI, THETA, PHI or THETA, PHI
    
    :Returns:
    
        out : tuple
              Depends on input: mirrored PSI, THETA, PHI 
              or THETA PHI
    '''
    
    if len(ang) == 2: return (ang[0], ang[1])
    else: return (180.0+ang[0], ang[1], ang[2])
    
def spider_euler(ang):
    ''' Conver Euler angle in ZYZ to proper SPIDER range
    
    :Parameters:
    
    ang : tuple
          Theta, PHI in degrees
    
    :Returns:
        
        theta : float
                0 <= theta < 90.0 or 180 <= theta < 270
        phi : float
              0 <= phi < 360.0
    '''
    
    if len(ang) == 3:
        psi, theta, phi = ang
        if theta < 180.0 and theta > 90.0:
            theta = 360.0 - theta
            phi += 180.0
            if phi > 360.0: phi-=360.0
        return psi, theta, phi
    else: raise ValueError, "Not implemented for other than 2 angles"

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
    

    ang = healpix.angles(resolution)
    #resolution = pow(2, resolution)
    if out is None: out=numpy.zeros((len(align), len(align[0])))
    cols = out.shape[1]
    for i in xrange(len(align)):
        #theta, phi = healpix.ensure_valid_deg(align[i,1], align[i,2], half)
        sp_i, sp_t, sp_p = spider_euler(align[i, :3])
        ipix = healpix.ang2pix(resolution, align[i,1], align[i,2], deg=True)
        rot = rotate_into_frame(ang[ipix], (sp_i, sp_t, sp_p))
        if half: ipix = healpix.ang2pix(resolution, align[i,1], align[i,2], deg=True, half=True)
        out[i, :3]=(rot, sp_t, sp_p)
        #out[i, :3]=(rot, align[i,2], sp_p)
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
    
    ang = healpix.angles(resolution)
    resolution = pow(2, resolution)
    if out is None: out=numpy.zeros((len(align), len(align[0])))
    cols = out.shape[1]
    for i in xrange(len(align)):
        sp_t, sp_p = spider_euler(align[i, 1:3])
        ipix = healpix.ang2pix(resolution, align[i,1], align[i,2], deg=True)
        rot, tx, ty = rotate_into_frame_2d(ang[ipix], sp_t, sp_p, align[i, 3], align[i,4], align[i,5])
        if half: ipix = healpix.ang2pix(resolution, align[i,1], align[i,2], deg=True, half=True)
        out[i, 1:6]=( sp_t, sp_p, rot, tx, ty)
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

def euler_to_vector(theta, phi):
    ''' Convert Euler angles to a vector
        
    :Parameters:
    
        theta : float
                Theta angle in degrees (psi,theta,phi = ZYZ)
        phi : float
              Phi angle in degrees (psi,theta,phi = ZYZ) 
    
    :Returns:
    
        x : float
            X-coordinate
        y : float
             Y-coordinate
        z : float
             Z-coordinate
    '''
    
    
    theta, phi = numpy.deg2rad((theta, phi))
    sintheta = numpy.sin(theta)
    return (sintheta*numpy.cos(phi), sintheta*numpy.sin(phi), numpy.cos(theta))


