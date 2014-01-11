''' Algorithms for one-class classification 

Also known as unary classification, this area attempts to find one class of objects among
all others. Applications include Anomaly detection, outlier detection and novelty detection.

.. Created on Jan 6, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import sklearn.covariance as skcov
import scipy.stats
import numpy

def mahalanobis_with_chi2(feat, prob_reject):
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

    try: robust_cov = skcov.MinCovDet().fit(feat)
    except: robust_cov = skcov.EmpiricalCovariance().fit(feat)
    dist = robust_cov.mahalanobis(feat - numpy.median(feat, 0))
    cut = scipy.stats.chi2.ppf(prob_reject, feat.shape[1])
    sel =  dist < cut
    return sel
