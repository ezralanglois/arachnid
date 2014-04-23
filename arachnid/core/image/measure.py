'''
.. Created on Apr 22, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
try:
    from skimage.filter import denoise_tv_chambolle as tv_denoise  #@UnresolvedImport
    tv_denoise;
except:
    from skimage.filter import tv_denoise  #@UnresolvedImport
from ..learn import distance
import scipy.ndimage
import numpy.linalg
import ndimage_interpolate
import ndimage_utility

def estimate_diameter(vol, cur_apix, apix=10, threshold=None, **extra):
    ''' Estimate the diameter of the object
    '''
    
    vol_sm = ndimage_interpolate.resample_fft_fast(vol, apix/cur_apix)
    vol_sm = tv_denoise(vol_sm, weight=10, eps=2.e-4, n_iter_max=200)
    apix=float(vol.shape[0])/vol_sm.shape[0]*cur_apix
    mask = ndimage_utility.tight_mask(vol_sm, threshold, 0, 0)[0]
    #ndimage_file.write_image('test03.mrc', mask)
    mask2 = scipy.ndimage.binary_dilation(mask, scipy.ndimage.generate_binary_structure(mask.ndim, 2), 1)
    coords = numpy.vstack(numpy.unravel_index(numpy.nonzero(mask.ravel()), mask.shape)).T.copy()
    diameter=distance.max_euclidiean_dist(coords)
    
    if 1 == 0:
        radii = getMinVolEllipse(coords)[1]
        print radii.max()/radii.min()
        
    return diameter*apix

def estimate_shape(vol, cur_apix, apix=20, threshold=None, **extra):
    ''' Get the basic shape of the object based on ellipsoid fitting
    
    :Parameters:
        
        vol : array
              Volume data
        cur_apix : float
                   Current pixel size of volume 
        apix : float
               Pixel size for downsampling
        threshold : float
                    Density threshold, default find automaticaly
        extra : dict
                Unused keyword arguments
    :Returns:
        
        radii : float
                Radii of minimum volumn ellipse scaled by pixel size
    '''
    
    vol_sm = ndimage_interpolate.resample_fft_fast(vol, apix/cur_apix)
    apix=float(vol.shape[0])/vol_sm.shape[0]*cur_apix
    vol_sm = tv_denoise(vol_sm, weight=10, eps=2.e-4, n_iter_max=200)
    mask = ndimage_utility.tight_mask(vol_sm, threshold, 0, 0)[0]
    coords = numpy.vstack(numpy.unravel_index(numpy.nonzero(mask.ravel()), vol_sm.shape)).T.copy()
    return minimum_volume_ellipse(coords)[1]*apix

def minimum_volume_ellipse(P=None, tolerance=0.01):
    ''' Find the minimum volume ellipsoid which holds all the points
    
    .. note::
        
        Adopted from:
        https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    '''
    
    (N, d) = numpy.shape(P)
    d = float(d)

    # Q will be our working array
    Q = numpy.vstack([numpy.copy(P.T), numpy.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * numpy.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = numpy.dot(Q, numpy.dot(numpy.diag(u), QT))
        M = numpy.diag(numpy.dot(QT , numpy.dot(numpy.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = numpy.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = numpy.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = numpy.dot(P.T, u)

    # the A matrix for the ellipse
    A = numpy.linalg.inv(
                   numpy.dot(P.T, numpy.dot(numpy.diag(u), P)) - 
                   numpy.array([[a * b for b in center] for a in center])
                   ) / d
                   
    # Get the values we'd like to return
    U, s, rotation = numpy.linalg.svd(A)
    radii = 1.0/numpy.sqrt(s)
    
    return (center, radii, rotation)

