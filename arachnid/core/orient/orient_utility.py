''' Utilities to convert between orientation parameters

.. Created on Sep 24, 2012
.. codeauthor:: robertlanglois
'''
import numpy
import transforms

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

def align_param_2D_to_3D(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
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
    
    if hasattr(rot, 'ndim') and rot.ndim > 0:
        raise ValueError, "Unsupported"
    else:
        R = transforms.euler_matrix(-rot, 0.0, 0.0, 'rzyz')
        T = transforms.translation_matrix((tx, ty, 0.0))
        M = numpy.dot(T, R)
    return -rot, M[3, 0], M[3, 1], M[3, 2]
    #

def align_param_2D_to_3D_simple(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
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
    
    ca = numpy.cos(-rot)
    sa = numpy.sin(-rot)
    sx = tx*ca + ty*sa
    sy = ty*ca - tx*sa
    return -rot, sx, sy