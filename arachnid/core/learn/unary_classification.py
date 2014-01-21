''' Algorithms for one-class classification 

Also known as unary classification, this area attempts to find one class of objects among
all others. Applications include Anomaly detection, outlier detection and novelty detection.

.. Created on Jan 6, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import sklearn.covariance as skcov
import scipy.stats
import numpy

def mahalanobis_with_chi2(feat, prob_reject, ret_dist=False):
    '''Reject outliers using one-class classification based on the mahalanobis distance
    estimate from a robust covariance as calculated by minimum covariance determinant.
    
    :Parameters:
    
    feat : array
           2D array where each row is a feature and each column a factor
    prob_reject : float
                  Probability threshold for rejecting outliers
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    sel : array
          Boolean selection array for each feature
    '''
    
    feat -= numpy.median(feat, axis=0)#feat.mean(axis=0)#scipy.stats.mstats.mode(feat, 0)[0]
    robust_cov = skcov.EmpiricalCovariance().fit(feat)
    dist = robust_cov.mahalanobis(feat)# - scipy.stats.mstats.mode(feat, 0)[0])
    cut = scipy.stats.chi2.ppf(prob_reject, feat.shape[1])
    sel =  dist < cut
    return (sel, dist) if ret_dist else sel

def robust_mahalanobis_with_chi2(feat, prob_reject, ret_dist=False):
    '''Reject outliers using one-class classification based on the mahalanobis distance
    estimate from a robust covariance as calculated by minimum covariance determinant.
    
    :Parameters:
    
    feat : array
           2D array where each row is a feature and each column a factor
    prob_reject : float
                  Probability threshold for rejecting outliers
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    sel : array
          Boolean selection array for each feature
    '''
    
    feat -= numpy.median(feat, axis=0)#feat.mean(axis=0)#scipy.stats.mstats.mode(feat, 0)[0]
    try: robust_cov = skcov.MinCovDet().fit(feat)
    except: robust_cov = skcov.EmpiricalCovariance().fit(feat)
    dist = robust_cov.mahalanobis(feat)# - scipy.stats.mstats.mode(feat, 0)[0])
    cut = scipy.stats.chi2.ppf(prob_reject, feat.shape[1])
    sel =  dist < cut
    return (sel, dist) if ret_dist else sel

def robust_euclidean(data, nsigma=2.67, mdata=None):
    ''' Robust outlier rejection using the MAD score
    
    :Parameters:
    
    data : array
           Sample vector for robust seleciton
    nsigma : float
             Number of standard deviation cutoff
    mdata : array, optional
            Array to calculate statistics over (alternative
            to `data`.
    
    :Returns:
    
    sel : array
          Boolean array of selected points
    '''
    
    if mdata is None: mdata=data
    cent = numpy.median(mdata, axis=0)#scipy.stats.mstats.mode(mdata, axis=0)[0]
    d = numpy.sqrt(numpy.sum(numpy.square(mdata-cent), axis=1))
    m = scipy.stats.mstats.mode(d)[0]
    s = robust_sigma(d)
    
    d = numpy.sqrt(numpy.sum(numpy.square(data-cent), axis=1))
    print m, s, m+s*nsigma, nsigma, numpy.max(d[d < m+s*nsigma]), numpy.min(d[d < m+s*nsigma])
    return d < m+s*nsigma

def robust_rejection(data, nsigma=2.67, mdata=None):
    ''' Robust outlier rejection using the MAD score
    
    :Parameters:
    
    data : array
           Sample vector for robust seleciton
    nsigma : float
             Number of standard deviation cutoff
    mdata : array, optional
            Array to calculate statistics over (alternative
            to `data`.
    
    :Returns:
    
    sel : array
          Boolean array of selected points
    '''
    
    if mdata is None: mdata=data
    m = numpy.median(mdata)
    s = robust_sigma(mdata)
    
    return data < m+s*nsigma

def robust_sigma(in_y, zero=0):
    """
   Calculate a resistant estimate of the dispersion of
   a distribution. For an uncontaminated distribution,
   this is identical to the standard deviation.

   Use the median absolute deviation as the initial
   estimate, then weight points using Tukey Biweight.
   See, for example, Understanding Robust and
   Exploratory Data Analysis, by Hoaglin, Mosteller
   and Tukey, John Wiley and Sons, 1983.

   .. note:: 
       
       ROBUST_SIGMA routine from IDL ASTROLIB.
       Python Code adopted from https://gist.github.com/1310949

   :History:
       * H Freudenreich, STX, 8/90
       * Replace MED call with MEDIAN(/EVEN), W. Landsman, December 2001
       * Converted to Python by P. L. Lim, 11/2009

   Examples:
   
   >>> result = robust_sigma(in_y, zero=1)

   :Parameters:
   
   in_y: array_like
         Vector of quantity for which the dispersion is
         to be calculated

   zero: int
          If set, the dispersion is calculated w.r.t. 0.0
          rather than the central value of the vector. If
          Y is a vector of residuals, this should be set.

   :Returns:
   
   out_val: float
            Dispersion value. If failed, returns -1.

    """
    # Flatten array
    y = in_y.reshape(in_y.size, )
    
    eps = 1.0E-20
    c1 = 0.6745
    c2 = 0.80
    c3 = 6.0
    c4 = 5.0
    c_err = -1.0
    min_points = 3
    
    if zero:
        y0 = 0.0
    else:
        y0 = numpy.median(y)
    
    dy    = y - y0
    del_y = abs( dy )
    
    # First, the median absolute deviation MAD about the median:
    
    mad = numpy.median( del_y ) / c1
    
    # If the MAD=0, try the MEAN absolute deviation:
    if mad < eps:
        mad = numpy.mean( del_y ) / c2
    if mad < eps:
        return 0.0
    
    # Now the biweighted value:
    u  = dy / (c3 * mad)
    uu = u*u
    q  = numpy.where(uu <= 1.0)
    count = len(q[0])
    if count < min_points:
        print 'ROBUST_SIGMA: This distribution is TOO WEIRD! Returning', c_err
        return c_err
    
    numerator = numpy.sum( (y[q]-y0)**2.0 * (1.0-uu[q])**4.0 )
    n    = y.size
    den1 = numpy.sum( (1.0-uu[q]) * (1.0-c4*uu[q]) )
    siggma = n * numerator / ( den1 * (den1 - 1.0) )
    
    if siggma > 0:
        out_val = numpy.sqrt( siggma )
    else:
        out_val = 0.0
    
    return out_val

def otsu(data, bins=0):
    ''' Otsu's threshold selection algorithm
    
    :Parameters:
        
    data : numpy.ndarray
           Data to find threshold
    bins : int
           Number of bins [if 0, use sqrt(len(data))]
    
    :Returns:
    
    th : float
         Optimal threshold to divide classes
    
    .. note::
        
        Code originally from:
            https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/cellprofiler/cpmath/otsu.py
    '''
        
    data = numpy.array(data).flatten()
    if bins <= 0: bins = int(numpy.sqrt(len(data)))
    if bins > len(data): bins = len(data)
    data.sort()
    var = running_variance(data)
    rvar = numpy.flipud(running_variance(numpy.flipud(data)))
    
    rng = len(data)/bins
    thresholds = data[1:len(data):rng]
    if 1 == 1:
        idx = numpy.arange(0,len(data)-1,rng, dtype=numpy.int)
        score_low = (var[idx] * idx)
        idx = numpy.arange(1,len(data),rng, dtype=numpy.int)
        score_high = (rvar[idx] * (len(data) - idx))
    else:
        score_low = (var[0:len(data)-1:rng] * numpy.arange(0,len(data)-1,rng))
        score_high = (rvar[1:len(data):rng] * (len(data) - numpy.arange(1,len(data),rng)))
    scores = score_low + score_high
    if len(scores) == 0: return thresholds[0]
    index = numpy.argwhere(scores == scores.min()).flatten()
    if len(index)==0: return thresholds[0]
    index = index[0]
    if index == 0: index_low = 0
    else: index_low = index-1
    if index == len(thresholds)-1: index_high = len(thresholds)-1
    else: index_high = index+1
    return (thresholds[index_low]+thresholds[index_high]) / 2

def running_variance(x, axis=None):
    '''Given a vector x, compute the variance for x[0:i]
    
    :Parameters:
        
    x : numpy.ndarray
        Sorted data
    axis : int, optional
           Axis along which the running variance is computed
    
    :Returns:
        
    var : numpy.ndarray
          Running variance
    
    .. note::
        
        Code originally from:
            https://svn.broadinstitute.org/CellProfiler/trunk/CellProfiler/cellprofiler/cpmath/otsu.py
    
        http://www.johndcook.com/standard_deviation.html
            S[i] = S[i-1]+(x[i]-mean[i-1])*(x[i]-mean[i])
            var(i) = S[i] / (i-1)
    '''
    n = len(x)
    # The mean of x[0:i]
    m = x.cumsum(axis=axis) / numpy.arange(1,n+1)
    # x[i]-mean[i-1] for i=1...
    x_minus_mprev = x[1:]-m[:-1]
    # x[i]-mean[i] for i=1...
    x_minus_m = x[1:]-m[1:]
    # s for i=1...
    s = (x_minus_mprev*x_minus_m).cumsum(axis=axis)
    var = s / numpy.arange(2,n+1)
    if axis is not None:
        var = numpy.mean(var, axis=0)
    # Prepend Inf so we have a variance for x[0]
    return numpy.hstack(([0],var))

