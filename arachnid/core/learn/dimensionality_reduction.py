''' Dimensionality reduction algorithms

This algorithms act on a matrix where each row is a sample and each column an observation to compress the number
of observations into a few highly informative factors.

.. Created on Jan 9, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging
import numpy
import core_utility
import scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def pca(trn, tst=None, frac=-1, mtrn=None, use_svd=True):
    ''' Principal component analysis using SVD
    
    :Parameters:
        
        trn : numpy.ndarray
              Matrix to decompose with PCA
        tst : numpy.ndarray
              Matrix to project into lower dimensional space (if not specified, then `trn` is projected)
        frac : float
               Number of Eigen vectors: frac < 1: fraction of variance, frac >= 1: number of components
    
    :Returns:
        
        val : numpy.ndarray
              Projected data
        idx : int
              Selected number of Eigen vectors
        V : numpy.ndarray
            Eigen vectors
        spec : float
               Explained variance
    '''

    
    if mtrn is None: mtrn = trn.mean(axis=0)
    trn = trn - mtrn

    if use_svd:
        U, d, V = scipy.linalg.svd(trn, False)
    else:
        d, V = numpy.linalg.eig(numpy.cov(trn))

    t = d**2/trn.shape[0]
    t /= t.sum()
    if frac >= 1:
        idx = int(frac)
    elif frac > 0.0:
        idx = numpy.sum(t.cumsum()<frac)+1
    else: idx = d.shape[0]
    if idx >= len(d): idx = 1
    if isinstance(tst, tuple):
        val = []
        for t in tst:
            t = t - mtrn
            val.append(d[:idx]*numpy.dot(V[:idx], tst.T).T)
        val = tuple(val)
    else:
        if tst is None: tst = trn
        else: tst = tst - mtrn
        val = d[:idx]*numpy.dot(V[:idx], tst.T).T
    return val, idx, V[:idx], numpy.sum(t[:idx])

def pca_fast(trn, tst=None, frac=0.0, centered=False):
    '''
    '''
    
    if not centered: 
        trn -= trn.mean(0)
    if trn.shape[1] <= trn.shape[0]:
        #C = numpy.cov(trn)
        #C = numpy.dot(trn.transpose(), trn)
        C = core_utility.fastdot_t1(trn, trn, None, 1.0/trn.shape[0])
    else:
        C = core_utility.fastdot_t2(trn, trn, None, 1.0/trn.shape[0])
        #C = numpy.dot(trn, trn.T)
        #numpy.multiply(C, 1.0/trn.shape[0], C)
    assert(numpy.alltrue(numpy.isfinite(C)))
    d, V = numpy.linalg.eigh(C)
    
    if trn.shape[1] > trn.shape[0]:
        #V = numpy.dot(trn.T, V)
        V = numpy.ascontiguousarray(V)
        assert(numpy.alltrue(numpy.isfinite(V)))
        V = core_utility.fastdot_t1(trn, V)
        
        err = numpy.seterr(divide='ignore', invalid='ignore')
        d = 1.0/numpy.sqrt(d)
        numpy.seterr(**err)
        d = numpy.where(numpy.isfinite(d), d, 0)
        V *= d
        assert(numpy.alltrue(numpy.isfinite(V)))
    d = numpy.where(d < 0, 0, d)
    tot = d.sum()
    if tot != 0.0: d /= tot
    idx = d.argsort()[::-1]
    d = d[idx]
    V = V[:, idx]
    
    if tst is not None:
        tst -= tst.mean(0)
        if frac >= 1:
            idx = int(frac)
        elif frac == 0.0:
            idx = d.shape[0]
        else:
            idx = numpy.sum(d.cumsum()<frac)+1
        #print V.dtype, tst.dtype
        val = core_utility.fastdot_t2(V[:, :idx].transpose().copy(), tst.astype(V.dtype)).T
        #val = numpy.dot(V[:, :idx].T, tst.T).T
        return V, d, val
    return V, d

def dhr_pca(trn, tst=None, neig=2, level=0.9, centered=False, iter=20):
    '''
    
    .. note ::
        
        Publication for algorithm: http://guppy.mpe.nus.edu.sg/~mpexuh/papers/DHRPCA-ICML.pdf
    '''
    
    level = int(trn.shape[0]*level)
    if not centered: 
        trn -= trn.mean(0)
        if tst is not None: tst -= tst.mean(0)
    if tst is None: tst=trn
    best = (0, None, None)
    wgt = numpy.ones(trn.shape[0])
    mat=None
    for i in xrange(iter):
        sel = wgt > 0
        mat = empirical_variance(trn, wgt, out=mat)
        eigvecs, eigvals, feat=pca_fast(mat, trn, neig, True)
        var = numpy.sum(numpy.square(feat), axis=1)[sel]
        
        tmp = 0.0
        for i in xrange(feat.shape[1]):
            val = numpy.square(feat[:, i])
            idx = numpy.argsort(val)
            tmp += numpy.sum(val[idx[:level]])/len(val)
        
        best_last=best[0]
        if tmp > best[0]: best = (tmp, eigvecs, eigvals)
        totw = numpy.sum(wgt)
        if var.shape[0] < 10 or totw < 10 or totw < (trn.shape[0]*0.1): break
        nu = numpy.min(1.0/var)
        _logger.debug("Opt: %g > %g (%d) -- nu: %f -- sum: %f"%(tmp, best_last, (tmp>best_last), nu, totw))
        tmp = nu*wgt[sel]*var
        wgt[sel] -= tmp
    
    if best[1] is None: return None, None
    V = best[1]
    eigv = best[2]
    feat = core_utility.fastdot_t2(V[:, :neig].transpose().copy(), tst.astype(V.dtype)).T
    return eigv, feat

def empirical_variance(feat1, weight=None, out=None):
    '''
    '''
    
    if weight is not None: 
        weight[weight<0]=0
        feat = feat1*numpy.sqrt(weight[:, numpy.newaxis])
    return core_utility.fastdot_t1(feat, feat, out, 1.0/feat.shape[0])

