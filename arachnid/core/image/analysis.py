''' Data analysis functions

.. todo:: Finish documenting analysis

.. todo:: Finish one_class_classification

.. Created on Jul 19, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

#from ..app import tracing
import numpy, scipy.special, scipy.linalg
from ..parallel import process_queue
from ..util import numpy_ext
import logging, functools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def robust_rejection(data, nsigma=2.67):
    ''' Robust outlier rejection
    '''
    
    m = numpy.median(data)
    s = robust_sigma(data)
    
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

   .. note:: ROBUST_SIGMA routine from IDL ASTROLIB.
   
   .. note:: Python Code from https://gist.github.com/1310949

   :History:
       * H Freudenreich, STX, 8/90
       * Replace MED call with MEDIAN(/EVEN), W. Landsman, December 2001
       * Converted to Python by P. L. Lim, 11/2009

   Examples
   --------
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
        if ref.ndim == 1: ref = ref.reshape((1, len(ref[0])))
        mat = scipy.spatial.distance.cdist(ref, data[i].reshape((1, len(data[i]))), metric='euclidean')
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
        total, length = process_queue.recreate_global_dense_matrix(data).shape
    else: total = len(process_queue.recreate_global_dense_matrix(data))
    sample, shmem_sample = process_queue.create_global_dense_matrix( ( sample_num, length )  )
    
    replace = sample_size == 0
    if sample_size == 0 : sample_size = total
    elif sample_size < 1.0: sample_size = int(sample_size*total)
    process_queue.map_array(_resample_worker, thread_count, shmem_sample, operator, data, sample_size, replace)
    return sample

def _resample_worker(beg, end, shmem_sample, operator, shmem_data, sample_size, replace, weight=None):
    ''' Resample the dataset and store in a subset
    
    :Parameters:
    
    beg : int
          Start of the sample range
    end : int
          End of the sample range
    shmem_sample : array
                   Array storing the samples 
    operator : function
               Generates a sample from a resampled distribution
    shmem_data : array
                 Array containing the data to resample
    sample_size : int
                  Size of the subset
    replace : bool
              Draw with replacement 
    weight : array
             Weight on each sample
    '''
    
    data = process_queue.recreate_global_dense_matrix(shmem_data)
    sample = process_queue.recreate_global_dense_matrix(shmem_sample)
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

def one_class_classification_old(data, nstd_min=3):
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
    
    dist = one_class_distance(data, nstd_min)
    dsel = one_class_selection(dist, 4)
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

def one_class_distance(data, nstd_min=3):
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
    
    sel = one_class_selection(data, nstd_min)
    axis = 0 if data.ndim == 2 else None
    m = numpy.median(data[sel], axis=axis)
    axis = 1 if data.ndim == 2 else None
    assert(data.ndim==2)
    dist = numpy.sqrt(numpy.sum(numpy.square(data-m), axis=axis)).squeeze()
    assert(dist.ndim==1)
    return dist

def one_class_selection(data, nstd_min=3):
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
    
    sel = numpy.ones(len(data), dtype=numpy.bool)    
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

