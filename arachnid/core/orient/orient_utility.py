''' Utilities to convert between orientation parameters

.. Created on Sep 24, 2012
.. codeauthor:: robertlanglois
'''
import numpy
import transforms,healpix
import logging, scipy, scipy.optimize
from ..parallel import process_queue
from ..image import rotate

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try:
    from core import _rotation_mapping
    _rotation_mapping;
except:
    from ..app import tracing
    tracing.log_import_error("Failed to import rotation mapping module - certain functionality will not be available", _logger)
    
def coarse_angles(resolution, align): # The bitterness of men who fear human progress
    '''
    TODO: disable mirror
    '''
    
    ang = healpix.angles(resolution)
    resolution = pow(2, resolution)
    new_ang=numpy.zeros((len(align), len(align[0])))
    for i in xrange(len(align)):
        theta=align[i,1]
        if align[i,1] > 180.0: theta -= 180.0
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(theta), numpy.deg2rad(align[i,2]))
        rang = rotate.rotate_euler(ang[ipix], (-align[i,3], theta, align[i,2]))
        rot = (rang[0]+rang[2])
        rt3d = align_param_2D_to_3D_simple(align[i, 3], align[i, 4], align[i, 5])
        rot, tx, ty = align_param_2D_to_3D_simple(rot, rt3d[1], rt3d[2])
        new_ang[i, 1:]=(align[i,1], align[i,2], rot, tx, ty)
        if len(align[0])>6: align[6] = ipix
    return new_ang

def coarse_angles2(resolution, align):
    '''
    TODO: disable mirror
    '''
    
    ang = healpix.angles(resolution)
    resolution = pow(2, resolution)
    new_ang=numpy.zeros((len(align), len(align[0])))
    for i in xrange(len(align)):
        theta=align[i,1]
        if align[i,1] > 180.0: theta -= 180.0
        ipix = healpix._healpix.ang2pix_ring(resolution, numpy.deg2rad(theta), numpy.deg2rad(align[i,2]))        
        
        refquat = spider_to_quaternion(ang[ipix])
        curquat = spider_to_quaternion((-align[i,3], theta, align[i,2]))
        curquat[1:] = -curquat[1:]
        rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(refquat, curquat), 'rzyz'))
        rot = rot[0]+rot[2]
        
        rt3d = align_param_2D_to_3D_simple(align[i, 3], align[i, 4], align[i, 5])
        rot, tx, ty = align_param_2D_to_3D_simple(rot, rt3d[1], rt3d[2])
        new_ang[i, 1:]=(align[i,1], align[i,2], rot, tx, ty)
        if len(align[0])>6: align[6] = ipix
    return new_ang

def rotation_from_euler(psi, theta, phi, axis='rzyz'):
    '''
    '''
    
    return transforms.rotation_from_matrix(transforms.euler_matrix(numpy.deg2rad(psi), numpy.deg2rad(theta), numpy.deg2rad(phi), axis))


def euler_to_vector(phi, theta):
    '''
    
    .. note::
        
        http://xmipp.cnb.csic.es/~xmipp/trunk/xmipp/documentation/html/geometry_8cpp_source.html
    
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

def fit_rotation(feat, maxfev=0):
    '''
    '''
    
    start = numpy.ones(9*feat.shape[1], dtype=feat.dtype)
    #
    map, success = scipy.optimize.leastsq(rotation_cost_function, start, args=(feat,), maxfev=maxfev)
    if (success > 4 or success < 1): 
        _logger.warn("Least-squares failed to find a solution: %d"%success)
    G = rotation_cost_function(map, feat)
    return map.reshape(3,3,feat.shape[1]), G

def rotation_cost_function(map, feat):
    ''' Mapping manifold to relative rotation
    '''
    
    G = numpy.zeros(feat.shape[0])
    _rotation_mapping.rotation_cost_function(map, G, feat)
    return G

def map_rotation(feat, map, orthogonal=True, iter=100, eps=1e-7):
    '''Map inhomogeneous manifold to a set of rotation matrices
    '''
    
    map = map.ravel()
    rot = numpy.zeros((feat.shape[0], 9), dtype=feat.dtype)
    if orthogonal:
        _rotation_mapping.map_orthogonal_rotations(feat, map, rot, iter, eps)
    else:
        _rotation_mapping.map_rotations(feat, map, rot)
    return rot

def map_rotation_py(feat, map, orthogonal=True):
    '''Map inhomogeneous manifold to a set of rotation matrices
    '''
    
    rot = numpy.zeros((feat.shape[0], 9), dtype=feat.dtype)
    map=map.reshape(9,feat.shape[1])
    assert(feat.shape[1] == 9)
    for i in xrange(feat.shape[0]):
        for j in xrange(9):
            rot[i, j] = numpy.dot(map[j], feat[i])
        if orthogonal:
            r = rot[i, :].reshape(3,3)
            U, s, Vh = scipy.linalg.svd(r)
            U = numpy.dot(U, Vh)
            rot[i, :] = U.ravel()
    return rot

def ensure_quaternion(angs, axes='rzyz', out=None):
    '''
    '''
    
    if out is None: out = numpy.zeros((angs.shape[0], 4))
    if angs.shape[1] == 9:
        M = numpy.identity(4)
        for i in xrange(angs.shape[0]):
            M[:3, :3] = angs[i, :].reshape((3,3))
            out[i, :] = transforms.quaternion_from_matrix(M)
    else:
        raise ValueError, "not implemented"
    return out

def orthogonalize(rot, out=None):
    '''Map inhomogeneous manifold to a set of rotation matrices
    '''
    
    if out is None: out=rot.copy()
    for i in xrange(rot.shape[0]):
        r = rot[i, :].reshape(3,3)
        U, s, Vh = scipy.linalg.svd(r)
        U = numpy.dot(U, Vh)
        out[i, :] = U.ravel()
    return out

#    for row, data in process_tasks.for_process_mp(iter_images(filename, label), image_processor, img1.shape, queue_limit=100, **extra):
#        mat[row, :] = data.ravel()[:img.shape[0]]

def optimal_inplace_rotation_mp(euler, row, col, worker_count=0, out=None):
    '''
    '''
    
    if worker_count  < 2:
        return optimal_inplace_rotation2(euler, row, col, out)
    if out is None: out = numpy.zeros(len(row))
    for i, d in process_queue.for_mapped(optimal_inplane_rotation_worker, worker_count, len(row), euler, row, col):
        out[i]=d
    return out

def convert_euler(psi,theta,phi, faxis='rzyz', taxis='rxyz'):
    '''
    '''
    
    return tuple(numpy.rad2deg(transforms.euler_from_matrix(transforms.euler_matrix(numpy.deg2rad(psi), numpy.deg2rad(theta), numpy.deg2rad(phi), faxis), taxis)))

def optimal_inplane_rotation_worker(beg, end, euler, row, col, process_number=None):
    '''
    '''
    
    for i in xrange(beg, end):
        refquat = spider_to_quaternion(euler[row[i]])
        curquat = spider_to_quaternion(euler[col[i]])
        curquat[1:] = -curquat[1:]
        rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(refquat, curquat), 'rzyz'))
        ang = rot[0]+rot[2]
        yield i, ang

def optimal_inplace_rotation2(euler, row, col, out=None):
    '''
    '''
    
    quat = spider_to_quaternion(euler)
    if out is None: out = numpy.zeros(len(row))
    for i in xrange(len(row)):
        refquat = quat[row[i]]
        curquat = quat[col[i]].copy()
        curquat[1:] = -curquat[1:]
        rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(refquat, curquat), 'rzyz'))
        out[i] = rot[0]+rot[2]
    return out

def optimal_inplace_rotation(refeuler, roteuler, out=None):
    '''
    '''
    
    rotquat = spider_to_quaternion(roteuler)
    refquat = spider_to_quaternion(refeuler)
    if out is None: out = numpy.zeros(len(rotquat))
    for i in xrange(len(out)):
        rotquat[i, 1:] = -rotquat[i, 1:]
        rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(refquat, rotquat[i]), 'rzyz'))
        out[i] = rot[0]+rot[2]
        '''
        ang = rot[0]+rot[2]
        if ang < 0.0: ang += 360.0
        if ang > 360.0: ang -= 360.0
        out[i]=numpy.fmod(ang, 360.0)
        '''
    return out

def optimal_inplace_rotation_old(refeuler, roteuler, out=None):
    '''
    '''
    
    rotquat = spider_to_quaternion(roteuler)
    refquat = spider_to_quaternion(refeuler)
    if out is None: out = numpy.zeros(len(rotquat))
    for i in xrange(len(out)):
        #rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(refquat, rotquat[i]), 'rzyz'))
        rot = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(rotquat[i], refquat), 'rzyz'))
        out[i]=-(rot[0]+rot[2])
    return out

def ensure_euler(rot, out=None, axes='rzyz'):
    '''
    '''
    
    if out is None: out = numpy.zeros((len(rot), 3))
    if rot.ndim == 2:
        if rot.shape[1]==3:
            out[:]=rot
        elif rot.shape[1]==4:
            for i in xrange(len(rot)):
                out[i, :] = transforms.euler_from_quaternion(rot[i], axes)
        elif rot.shape[1]==9:
            for i in xrange(len(rot)):
                out[i, :] = transforms.euler_from_matrix(rot[i].reshape((3,3)))
        else: raise ValueError, "Not supported"
    else: raise ValueError, "Not supported"
    return out

def spider_to_quaternion(euler, out=None):
    ''' Convert SPIDER rotations to quaternions
    
    :Parameters:
    
    euler : array
            Array of euler angles
    out : array, optional
           Array of quaternions
    
    :Returns:
    
    out : array
           Array of quaternions
    '''
    
    if len(euler) == 3 and (not hasattr(euler[0], '__len__') or len(euler[0])==1):
        euler1 = numpy.deg2rad(euler[:3])
        return transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
    
    if out is None: out = numpy.zeros((len(euler), 4))
    for i in xrange(out.shape[0]):
        euler1 = numpy.deg2rad(euler[i, :3])
        out[i, :] = transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
    return out

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

def align_param_3D_to_2D_simple(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
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
    #;x24= x22*COS(-x35)-x23*SIN(-x35)   ; no inversion
    #;x25= x22*SIN(-x35)+x23*COS(-x35)
    
    rot1 = numpy.deg2rad(rot)
    ca = numpy.cos(rot1)
    sa = numpy.sin(rot1)
    sx = tx*ca - ty*sa
    sy = tx*sa + ty*ca
    return -rot, sx, sy
    

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
    #x24=  x22*COS(-x35)+x23*SIN(-x35)  ;with matrix inversion
    #x25= -x22*SIN(-x35)+x23*COS(-x35)
    rot1 = -numpy.deg2rad(rot)
    ca = numpy.cos(rot1)
    sa = numpy.sin(rot1)
    sx =  tx*ca + ty*sa
    sy = -tx*sa + ty*ca 
    return -rot, sx, sy

def compute_frames_reject(search_grid=40.0, **extra):
    '''
    '''
    
    start, stop, step = 0.0, 1.0, search_grid
    step = float(stop-start)/step
    frames = []
    for i in numpy.arange(start, stop, step):
        for j in numpy.arange(start, stop, step):
            for k in numpy.arange(start, stop, step):
                w = i*i+j*j+k*k
                if w < 1.0:
                    frames.append( (i, j, k, numpy.sqrt(1.0-w)) )
                    frames.append( (i, j, k,-numpy.sqrt(1.0-w)) )
                    frames.append( (-i, -j, -k, numpy.sqrt(1.0-w)) )
                    frames.append( (-i, -j, -k,-numpy.sqrt(1.0-w)) )
    return frames

