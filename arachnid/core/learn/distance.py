''' Distance routines

.. Created on Apr 22, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import core_utility
import logging
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def max_euclidiean_dist(samp, batch=10000):
    '''
    '''
    
    dtype = samp.dtype
    if callable(batch): batch = batch(dtype)
    nbatch = max(1, int(samp.shape[0]/float(batch)))
    batch = int(samp.shape[0]/float(nbatch))
    dense = numpy.empty((batch,batch), dtype=dtype)
    a = (samp**2).sum(axis=1)
    maxval = -1e20
    for r in xrange(0, samp.shape[0], batch):
        rnext = min(r+batch, samp.shape[0])
        s1 = samp[r:rnext]
        for c in xrange(0, samp.shape[0], batch):
            s2 = samp[c:min(c+batch, samp.shape[0])]
            tmp = dense.ravel()[:s1.shape[0]*s2.shape[0]].reshape((s1.shape[0],s2.shape[0]))
            
            core_utility.fastdot_t2(s1, s2, tmp, -2.0)
            
            dist2=tmp
            dist2 += a[c:c+batch]#, numpy.newaxis]
            dist2 += a[r:rnext, numpy.newaxis]
            val=dist2.max()
            if val > maxval: maxval = val
    return numpy.sqrt(maxval)