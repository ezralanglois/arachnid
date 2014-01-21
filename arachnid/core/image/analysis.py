''' Data analysis functions

.. todo:: Finish documenting analysis

.. todo:: Finish one_class_classification

.. Created on Jul 19, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

#from ..app import tracing
import numpy
import scipy.special
import scipy.linalg
import scipy.spatial
from ..parallel import process_queue
from ..util import numpy_ext
import logging, functools
import manifold

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def robust_rejection(data, nsigma=2.67, mdata=None):
    ''' Robust outlier rejection
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

def subset_no_overlap(data, overlap, n=100):
    ''' Select a non-overlapping subset of the data based on hyper-sphere exclusion
    
    :Parameters:
    
    data : array
           Full set array
    overlap : float
              Exclusion distance
    n : int
        Maximum number in the subset
    
    :Returns:
    
    out : array
          Indices of non-overlapping subset
    '''
    
    k=1
    out = numpy.zeros(n, dtype=numpy.int)
    for i in xrange(1, len(data)):
        ref = data[out[:k]]
        if ref.ndim == 1: ref = ref.reshape((1, data.shape[1]))
        mat = scipy.spatial.distance.cdist(ref, data[i].reshape((1, len(data[i]))), metric='euclidean')
        if mat.ndim == 0 or mat.shape[0]==0: continue
        val = numpy.min(mat)
        if val > overlap:
            out[k] = i
            k+=1
            if k >= n: break
    return out[:k]

def resample(data, sample_num, sample_size, thread_count=0, operator=functools.partial(numpy.mean, axis=0), length=None):
    ''' Resample a dataset and apply the operator functor to each sample. The result is stored in
    a 2D array.
    
    :Parameters:
    
    data : array
           Data array
    sample_num : int
                 Number of times to resample
    sample_size : int
                  Number of random samples to draw 
    thread_count : int
                   Number of threads
    operator : function
               Function to apply to each random sample (Default mean operator)
    length : int
             Size of return result from operator
    
    :Returns:
    
    sample : array
             Result of operator over each sample
    '''
    
    if length is None: 
        total, length = data.shape
    else: total = len(data)
    sample = numpy.zeros(( sample_num, length ))
    
    replace = sample_size == 0
    if sample_size == 0 : sample_size = total
    elif sample_size < 1.0: sample_size = int(sample_size*total)
    process_queue.map_array(_resample_worker, thread_count, sample, operator, data, sample_size, replace)
    return sample

def _resample_worker(beg, end, sample, operator, data, sample_size, replace, weight=None):
    ''' Resample the dataset and store in a subset
    
    :Parameters:
    
    beg : int
          Start of the sample range
    end : int
          End of the sample range
    sample : array
                   Array storing the samples 
    operator : function
               Generates a sample from a resampled distribution
    data : array
                 Array containing the data to resample
    sample_size : int
                  Size of the subset
    replace : bool
              Draw with replacement 
    weight : array
             Weight on each sample
    '''
    
    index = numpy.arange(data.shape[0], dtype=numpy.int)
    for i in xrange(beg, end):
        selected = numpy_ext.choice(index.copy(), size=sample_size, replace=replace, p=weight)
        subset = data[selected].squeeze()
        try:
            sample[i, :] = operator(subset)
        except:
            _logger.error("%d > %d --- %d"%(i, len(sample), end))
            raise

def pca_train(trn, frac=-1, mtrn=None):
    ''' Principal component analysis using SVD
    
    :Parameters:
        
    trn : numpy.ndarray
          Matrix to decompose with PCA
    tst : numpy.ndarray
          Matrix to project into lower dimensional space (if not specified, then `trn` is projected)
    frac : float
           Number of Eigen vectors: frac < 0: fraction of variance, frac >= 1: number of components, frac == 0, automatically assessed
    
    :Returns:
        
    val : numpy.ndarray
          Projected data
    idx : int
          Selected number of Eigen vectors
    V : numpy.ndarray
        Eigen vectors
    spec : float
           Explained variance
               
    .. note::
        
        Automatic Assessment is performed using an algorithm by `Thomas P. Minka:
        Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
        
        Code originally from:
        `https://raw.github.com/scikit-learn/scikit-learn/master/sklearn/decomposition/pca.py`
    '''
    
    use_svd=True
    if mtrn is None: mtrn = trn.mean(axis=0)
    trn = trn - mtrn

    if use_svd:
        U, d, V = scipy.linalg.svd(trn, False)
    else:
        d, V = numpy.linalg.eig(numpy.corrcoef(trn, rowvar=0))

    t = d**2/trn.shape[0]
    t /= t.sum()
    if frac >= 1:
        idx = int(frac)
    elif frac == 0.0:
        if 1 == 0:
            diff = numpy.abs(d[:len(d)-1]-d[1:])
            idx = diff.argmax()+1
        else:
            idx = _assess_dimension(t, trn.shape[0], trn.shape[1])+1
    elif frac > 0.0:
        idx = numpy.sum(t.cumsum()<frac)+1
    else: idx = d.shape[0]
    #_logger.error("pca: %s -- %s"%(str(V.shape), str(idx)))
    if idx >= len(d): idx = 1
    return idx, V[:idx], numpy.sum(t[:idx])

def pca_fast(trn, tst=None, frac=0.0, centered=False):
    '''
    '''
    
    if not centered: 
        trn -= trn.mean(0)
    if trn.shape[1] <= trn.shape[0]:
        #C = numpy.cov(trn)
        #C = numpy.dot(trn.transpose(), trn)
        C = manifold.fastdot_t1(trn, trn, None, 1.0/trn.shape[0])
    else:
        C = manifold.fastdot_t2(trn, trn, None, 1.0/trn.shape[0])
        #C = numpy.dot(trn, trn.T)
        #numpy.multiply(C, 1.0/trn.shape[0], C)
    assert(numpy.alltrue(numpy.isfinite(C)))
    d, V = numpy.linalg.eigh(C)
    
    if trn.shape[1] > trn.shape[0]:
        #V = numpy.dot(trn.T, V)
        V = numpy.ascontiguousarray(V)
        assert(numpy.alltrue(numpy.isfinite(V)))
        V = manifold.fastdot_t1(trn, V)
        
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
        val = manifold.fastdot_t2(V[:, :idx].transpose().copy(), tst.astype(V.dtype)).T
        #val = numpy.dot(V[:, :idx].T, tst.T).T
        return V, d, val
    return V, d

def dhr_pca(trn, tst=None, neig=2, level=0.9, centered=False, iter=20):
    '''
    http://guppy.mpe.nus.edu.sg/~mpexuh/papers/DHRPCA-ICML.pdf
    '''
    
    #_logger.error("trn: %s"%str(trn.shape))
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
        #assert(trn.shape[0]==mat.shape[0])
        eigvecs, eigvals, feat=pca_fast(mat, trn, neig, True)
        assert(feat.shape[0]==trn.shape[0])
        #_logger.error("trn: %s -- %s"%(str(trn.shape), str(feat.shape)))
        var = numpy.sum(numpy.square(feat), axis=1)[sel]
        
        tmp = 0.0
        for i in xrange(feat.shape[1]):
            val = numpy.square(feat[:, i])
            idx = numpy.argsort(val)
            tmp += numpy.sum(val[idx[:level]])/len(val)
        
        
        best_last=best[0]
        if tmp > best[0]: best = (tmp, eigvecs, eigvals)
        #else: break
        totw = numpy.sum(wgt)
        if var.shape[0] < 10 or totw < 10 or totw < (trn.shape[0]*0.1): break
        nu = numpy.min(1.0/var)
        _logger.info("Opt: %g > %g (%d) -- nu: %f -- sum: %f"%(tmp, best_last, (tmp>best_last), nu, totw))
        tmp = nu*wgt[sel]*var
        wgt[sel] -= tmp
    
    if best[1] is None: return None, None
    V = best[1]
    eigv = best[2]
    feat = manifold.fastdot_t2(V[:, :neig].transpose().copy(), tst.astype(V.dtype)).T
    #_logger.error("%s == %s | %s %s"%(str(feat.shape), str(trn.shape), str(V.shape), str(tst.shape)))
    assert(feat.shape[0] == trn.shape[0])
    return eigv, feat

def empirical_variance(feat1, weight=None, out=None):
    '''
    '''
    
    if weight is not None: 
        weight[weight<0]=0
        feat = feat1*numpy.sqrt(weight[:, numpy.newaxis])
    return manifold.fastdot_t1(feat, feat, out, 1.0/feat.shape[0])

def empirical_variance_slow(feat, weight=None, out=None):
    '''
    '''
    
    if out is None: out = numpy.zeros((feat.shape[1], feat.shape[1]), dtype=feat.dtype)
    else: out[:]=0
    
    if weight is not None:
        for i in xrange(feat.shape[0]):
            tmp = feat[i].reshape((1, feat.shape[1]))
            manifold.fastdot_t1(tmp, tmp, out, weight[i], 1.0)
            #out += manifold.fastdot_t1(feat, feat)*weight[i]
            #out += numpy.dot(feat.T, feat)*weight[i]
    else:
        for i in xrange(feat.shape[0]):
            manifold.fastdot_t1(feat, feat, out, 1.0, 1.0)
    out /= feat.shape[0]
    return out

'''
import numpy

def pca_svd(flat):
   u, s, vt = numpy.linalg.svd(flat, full_matrices = 0)
   pcs = vt
   v = numpy.transpose(vt)
   data_count = len(flat)
   variances = s**2 / data_count
   positions =  u * s
   return pcs, variances, positions

def pca_eig(flat):
   values, vectors = _symm_eig(flat)
   pcs = vectors.transpose()
   variances = values / len(flat)
   positions = numpy.dot(flat, vectors)
   return pcs, variances, positions

def _symm_eig(a):
   """Given input a, return the non-zero eigenvectors and eigenvalues  
of the symmetric matrix a'a.

   If a has more columns than rows, then that matrix will be rank- 
deficient,
   and the non-zero eigenvalues and eigenvectors of a'a can be more  
easily extracted
   from the matrix aa'. From the properties of the SVD:
     if a of shape (m,n) has SVD u*s*v', then:
       a'a = v*s's*v'
       aa' = u*ss'*u'
     let s_hat, an array of shape (m,n), be such that s * s_hat = I(m,m)
     and s_hat * s = I(n,n). Thus, we can solve for u or v in terms of  
the other:
       v = a'*u*s_hat'
       u = a*v*s_hat
   """
   m, n = a.shape
   if m >= n:
     # just return the eigenvalues and eigenvectors of a'a
     vecs, vals = _eigh(numpy.dot(a.transpose(), a))
     vecs = numpy.where(vecs < 0, 0, vecs)
     return vecs, vals
   else:
     # figure out the eigenvalues and vectors based on aa', which is  
smaller
     sst_diag, u = _eigh(numpy.dot(a, a.transpose()))
     # in case due to numerical instabilities we have sst_diag < 0  
anywhere,
     # peg them to zero
     sst_diag = numpy.where(sst_diag < 0, 0, sst_diag)
     # now get the inverse square root of the diagonal, which will  
form the
     # main diagonal of s_hat
     err = numpy.seterr(divide='ignore', invalid='ignore')
     s_hat_diag = 1/numpy.sqrt(sst_diag)
     numpy.seterr(**err)
     s_hat_diag = numpy.where(numpy.isfinite(s_hat_diag), s_hat_diag, 0)
     # s_hat_diag is a list of length m, a'u is (n,m), so we can just  
use
     # numpy's broadcasting instead of matrix multiplication, and only  
create
     # the upper mxm block of a'u, since that's all we'll use anyway...
     v = numpy.dot(a.transpose(), u[:,:m]) * s_hat_diag
     return sst_diag, v

def _eigh(m):
   values, vectors = numpy.linalg.eigh(m)
   order = numpy.flipud(values.argsort())
   return values[order], vectors[:,order]
'''

def pca(trn, tst=None, frac=-1, mtrn=None, use_svd=True):
    ''' Principal component analysis using SVD
    
    :Parameters:
        
    trn : numpy.ndarray
          Matrix to decompose with PCA
    tst : numpy.ndarray
          Matrix to project into lower dimensional space (if not specified, then `trn` is projected)
    frac : float
           Number of Eigen vectors: frac < 0: fraction of variance, frac >= 1: number of components, frac == 0, automatically assessed
    
    :Returns:
        
    val : numpy.ndarray
          Projected data
    idx : int
          Selected number of Eigen vectors
    V : numpy.ndarray
        Eigen vectors
    spec : float
           Explained variance
               
    .. note::
        
        Automatic Assessment is performed using an algorithm by `Thomas P. Minka:
        Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
        
        Code originally from:
        `https://raw.github.com/scikit-learn/scikit-learn/master/sklearn/decomposition/pca.py`
    '''

    
    if mtrn is None: mtrn = trn.mean(axis=0)
    trn = trn - mtrn

    if use_svd:
        U, d, V = scipy.linalg.svd(trn, False)
    else:
        d, V = numpy.linalg.eig(numpy.cov(trn))
        #corrcoef

    t = d**2/trn.shape[0]
    t /= t.sum()
    if frac >= 1:
        idx = int(frac)
    elif frac == 0.0:
        if 1 == 0:
            diff = numpy.abs(d[:len(d)-1]-d[1:])
            idx = diff.argmax()+1
        else:
            idx = _assess_dimension(t, trn.shape[0], trn.shape[1])+1
    elif frac > 0.0:
        idx = numpy.sum(t.cumsum()<frac)+1
    else: idx = d.shape[0]
    #_logger.error("pca: %s -- %s"%(str(V.shape), str(idx)))
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
    #_logger.error("pca2: %s -- %s"%(str(V.shape), str(tst.shape)))
    return val, idx, V[:idx], numpy.sum(t[:idx])

def _assess_dimension(spectrum, n_samples, n_features):
    '''Compute the likelihood of a rank ``rank`` dataset

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``.

    :Parameters:
    
    spectrum : numpy.ndarray
              Data spectrum
    n_samples : int
               Number of samples
    dim : int
         Embedding/empirical dimension

    :Returns:
    
    ll : float,
        The log-likelihood

    .. note::
        
        This implements the method of `Thomas P. Minka:
        Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`
        
        Code originally from:
        https://raw.github.com/scikit-learn/scikit-learn/master/sklearn/decomposition/pca.py
    
    '''
    
    max_ll = (0, 0)
    for rank in xrange(len(spectrum)):
        pu = -rank * numpy.log(2)
        for i in xrange(rank):
            pu += (scipy.special.gammaln((n_features - i) / 2) - numpy.log(numpy.pi) * (n_features - i) / 2)
    
        pl = numpy.sum(numpy.log(spectrum[:rank]))
        pl = -pl * n_samples / 2
    
        if rank == n_features:
            pv = 0
            v = 1
        else:
            v = numpy.sum(spectrum[rank:]) / (n_features - rank)
            pv = -numpy.log(v) * n_samples * (n_features - rank) / 2
    
        m = n_features * rank - rank * (rank + 1) / 2
        pp = numpy.log(2 * numpy.pi) * (m + rank + 1) / 2
    
        pa = 0
        spectrum_ = spectrum.copy()
        spectrum_[rank:n_features] = v
        for i in xrange(rank):
            for j in xrange(i + 1, len(spectrum)):
                pa += (numpy.log((spectrum[i] - spectrum[j]) * (1. / spectrum_[j] - 1. / spectrum_[i])) + numpy.log(n_samples))
    
        ll = pu + pl + pv + pp - pa / 2 - rank * numpy.log(n_samples) / 2
        if numpy.isfinite(ll) and ll > max_ll[0]: max_ll = (ll, rank)

    return max_ll[1]

def one_class_classification_old(data, nstd_min=3, sel=None):
    ''' Classify a set of data into one-class and outliers
    
    :Parameters:
        
    data : numpy.ndarray
           Data to find center
    nstd_min : int
               Minimum number of standard deviations for outlier rejection

    :Returns:
    
    dist : numpy.ndarray
          Distance of each point in data from the center
    '''
    
    dist = one_class_distance(data, nstd_min, sel)
    dsel = one_class_selection(dist, 4, sel)
    #th = otsu(dist, len(dist)/16)
    #th = otsu(dist, numpy.sqrt(len(dist)))
    #dsel = dist < th
    return dsel

def one_class_classification(data_reduced,raveled_im,plot):
    # rows = number of datapoints
    # return index of cut
    row, col = numpy.shape(data_reduced) 
    dist = one_class_distance(data_reduced)
    sind = numpy.argsort(dist)
    c1v = numpy.zeros(row)
    c2v = c1v
    for n in numpy.arange(0,row-1):
        m = min(sind)
        l = numpy.nonzero(sind == m)
        sind[l] = row+2
        class1 = numpy.nonzero(sind == (row+2))
        class2 = numpy.nonzero(sind != (row+2))
        c1v[n] = numpy.mean(numpy.std(raveled_im[class1,:])**2)
        c2v[n] = numpy.mean(numpy.std(raveled_im[class2,:])**2)
    c2vscaled = c2v/max(c2v)
    xscaled = numpy.arange(0,row,dtype=float)/row
    d = numpy.sqrt((1-xscaled)**2+(c2vscaled)**2)
    m = min(d)
    cut = d == m #numpy.nonzero(d == m)
    '''
    if (plot == 1) and (pylab is not None) :
        pylab.plot(xscaled,c2vscaled)
        pylab.show()
    '''
    return cut

def one_class_distance(data, nstd_min=3, sel=None):
    ''' Calculate the distance from the median center of the data
    
    :Parameters:
        
    data : numpy.ndarray
           Data to find center
    nstd_min : int
               Minimum number of standard deviations for outlier rejection

    :Returns:
    
    dist : numpy.ndarray
          Distance of each point in data from the center
    '''
    
    sel = one_class_selection(data, nstd_min, sel)
    axis = 0 if data.ndim == 2 else None
    m = numpy.median(data[sel], axis=axis)
    axis = 1 if data.ndim == 2 else None
    assert(data.ndim==2)
    dist = numpy.sqrt(numpy.sum(numpy.square(data-m), axis=axis)).squeeze()
    assert(dist.ndim==1)
    return dist

def one_class_selection(data, nstd_min=3, sel=None):
    ''' Select a set of non-outlier projections
    
    This code starts at 10 (max) standard deviations from the median
    and goes down to `nstd_min`, throwing out outliers and recalculating
    the median and standard deviation.
    
    :Parameters:
        
    data : numpy.ndarray
           Data to find non-outliers
    nstd_min : int
               Minimum number of standard deviations

    :Returns:

    selected : numpy.ndarray
               Boolen array of selected data
    '''
    
    assert(hasattr(data, 'ndim'))
    axis = 0 if data.ndim == 2 else None
    m = numpy.median(data, axis=axis)
    s = numpy.std(data, axis=axis)
    maxval = numpy.max(data, axis=axis)
    minval = numpy.min(data, axis=axis)
    
    if sel is None: sel = numpy.ones(len(data), dtype=numpy.bool)    
    start = int(min(10, max(numpy.max((maxval-m)/s),numpy.max((m-minval)/s))))
    for nstd in xrange(start, nstd_min-1, -1):
        m=numpy.median(data[sel], axis=axis)
        s = numpy.std(data[sel], axis=axis)
        hcut = m+s*nstd
        lcut = m-s*nstd
        
        if data.ndim > 1:
            tmp = data<hcut
            assert(tmp.shape[1]==data.shape[1])
            sel = numpy.logical_and(data[:, 0]<hcut[0], data[:, 0]>lcut[0])
            for i in xrange(1, data.shape[1]):
                sel = numpy.logical_and(sel, numpy.logical_and(data[:, i]<hcut[i], data[:, i]>lcut[i]))
        else:
            sel = numpy.logical_and(data<hcut, data>lcut)
        assert(numpy.sum(sel)>0)
    return sel

def threshold_max(data, threshold, max_num, reverse=False):
    ''' Ensure no more than `max_num` data points are selected by
    the given threshold.
    
    :Parameters:
    
    data : array
           Array of values to select
    threshold : float
                Values greater than thresold are selected
    max_num : int
              Only the `max_num` highest are selected
    reverse : bool
              Reverse the order of selection
    
    :Returns:
    
    threshold : float
                New threshold to use
    '''
    
    idx = numpy.argsort(data)
    if not reverse: idx = idx[::-1]
    threshold = numpy.searchsorted(data[idx], threshold)
    if threshold > max_num: threshold = max_num
    try:
        return data[idx[threshold]]
    except:
        _logger.error("%d > %d -> %d"%(threshold, idx.shape[0], max_num))
        raise

def threshold_from_total(data, total, reverse=False):
    ''' Find the threshold given the total number of elements kept.
    
    :Parameters:
    
    data : array
           Array of values to select
    total : int or float
            Number of elements to keep (if float and less than 1.0, then fraction)
    reverse : bool
              Reverse the order of selection
    
    :Returns:
    
    threshold : float
                New threshold to use
    '''
    
    if total < 1.0: total = int(total*data.shape[0])
    idx = numpy.argsort(data)
    if not reverse: idx = idx[::-1]
    try:
        return data[idx[total]]
    except:
        _logger.error("%d > %d"%(total, idx.shape[0]))
        raise

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

def online_variance(data, axis=None):    
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
        
        Adopted from two-pass algorithm:
            http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''
    
    if axis is None: 
        axis = 0
        data = data.ravel()
    meanx = data.cumsum(axis)
    meanx /= numpy.arange(1, data.shape[axis]+1).reshape(data.shape[axis], 1)
    out = numpy.cumsum(numpy.square(data-meanx), axis=axis)
    return out/(len(data)-1)

