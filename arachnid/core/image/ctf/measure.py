'''
.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from .. import ndimage_utility
import numpy, scipy.linalg
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def estimate_circularity(pow, offset, end, ma_len=27):
    ''' Estimate the circularity of a 2D power spectra
    
    :Parameters:
    
    pow : array
          2D power spectra
    offset : int
             Start of range to test
    end : int
          End of range to test
    ma_len : int
             Lenght of moving average window
          
    :Returns:
    
    score : float
            Circularity score
    '''
    
    lp_pow, log_base = ndimage_utility.logpolar(numpy.log(pow+1))
    try:
        offset = int(numpy.log(offset)/numpy.log(log_base))
    except: return 0
    end = int(numpy.log(end)/numpy.log(log_base))
    lp_pow = lp_pow[:, offset:end]
    if lp_pow.shape[1] == 0: return -1.0
    moo = lp_pow.mean(0)
    bg = ndimage_utility.moving_average(moo, ma_len)
    moo -= bg
    segavg = lp_pow
    segavg -= bg
    trn = segavg - moo
    d = scipy.linalg.svd(trn, False)[1]
    return d[1]/d[0]

def estimate_circularity_bfac(pow, offset, end, ma_len=27):
    ''' Estimate the circularity of a 2D power spectra
    
    :Parameters:
    
    pow : array
          2D power spectra
    offset : int
             Start of range to test
    end : int
          End of range to test
    ma_len : int
             Lenght of moving average window
          
    :Returns:
    
    score : float
            Circularity score
    '''
    
    pow = pow.copy()
    pow -= pow.min()
    assert(numpy.alltrue((pow+1)>0))
    lp_pow, log_base = ndimage_utility.logpolar(numpy.log(pow+1))
    try:
        offset = int(numpy.log(offset)/numpy.log(log_base))
    except: return 0
    end = int(numpy.log(end)/numpy.log(log_base))
    lp_pow = lp_pow[:, offset:end]
    if lp_pow.shape[1] == 0: return -1.0
    moo = lp_pow.mean(0)
    bg = ndimage_utility.moving_average(moo, ma_len)
    moo -= bg
    segavg = lp_pow
    segavg -= bg
    
    normalize(moo)
    zero = numpy.mean(moo[len(moo)-len(moo)/5:])
    maxima=find_extrema(moo, moo>zero)
    try:
        minima=find_extrema(moo, moo<zero, numpy.argmin)
    except:
        import logging
        logging.error("Zero: %f"%zero)
        raise
    
    invert_curve(moo)
    scale_extrema(moo, maxima)
    invert_curve(moo)
    scale_extrema(moo, minima)
    
    segavg = local_average(segavg, 32)
    
    for i in xrange(segavg.shape[0]):
        normalize(segavg[i])
        zero = numpy.mean(segavg[i][len(segavg[i])-len(segavg[i])/5:])
        maxima=find_extrema(segavg[i], segavg[i]>zero)
        minima=find_extrema(moo, moo<zero, numpy.argmin)
        invert_curve(segavg[i])
        scale_extrema(segavg[i], maxima)
        invert_curve(segavg[i])
        scale_extrema(segavg[i], minima)
    
    trn = segavg - moo
    d = scipy.linalg.svd(trn, False)[1]
    return d[1]/d[0]

def local_average(pow, total=32, axis=0):
    ''' Locally average sectors in a polar transformed image
    
    :Parameters:
    
    pow : array
          Polar transformed matrix
          
    total : int
            Total number of sectors
    
    :Returns:
    
    avg : array
          Averaged radial lines in polar matrix
    '''
    
    win = pow.shape[0]/total
    segavg = numpy.zeros((pow.shape[0]-win, pow.shape[1]))
    b = win/2
    for i in xrange(segavg.shape[0]):
        e = b + 1
        segavg[i, :] = pow[b:e].mean(axis) #-bg
        b=e
    return segavg


def background_correct(roo, peak_snr=1, peak_rng=[1,10], **extra):
    '''
    '''
    
    roo=roo.copy()
    import scipy.signal
    maxima1 = scipy.signal.find_peaks_cwt(roo, numpy.arange(*peak_rng), min_snr=peak_snr)
    invert_curve(roo)
    minima1 = scipy.signal.find_peaks_cwt(roo, numpy.arange(*peak_rng), min_snr=peak_snr)
    
    i, j=0, 0
    maxima=[]
    minima=[]
    _logger.error("here1: %d, %d"%(len(maxima1), len(minima1)))
    maxima1.sort()
    minima1.sort()
    while i < len(maxima1) and j < len(minima1):
        _logger.error("loop1: %d, %d"%(i, j))
        while i < len(maxima1) and j < len(minima1) and maxima1[i] < minima1[j]: i+=1
        _logger.error("loop2: %d, %d"%(i, j))
        if i > 0: maxima.append(maxima1[i-1])
        while i < len(maxima1) and j < len(minima1) and maxima1[i] >= minima1[j]: j+=1
        _logger.error("loop3: %d, %d"%(i, j))
        if j > 0: minima.append(minima1[j-1])
    _logger.error("here2: %d, %d -- %d"%(len(maxima), len(minima), len(set(minima))))
    maxima = list(set(maxima))
    minima = list(set(minima))
    maxima.sort()
    minima.sort()
    
    #invert_curve(roo)
    scale_extrema(roo, maxima)
    invert_curve(roo)
    scale_extrema(roo, minima)
    #invert_curve(roo)
    return roo

def factor_correction(roo, beg, end):
    '''
    '''
    
    roo = normalize(roo.copy())
    if 1 == 0:
        roo = roo.copy()
        roo -= roo.min()
        roo /= roo.max()
    zero = numpy.mean(roo[len(roo)-len(roo)/5:])
    roo1=roo[beg:end].copy()
    
    if 1 == 1:
        maxima=find_extrema(roo1, roo1>zero)
        minima=find_extrema(roo1, roo1<zero, numpy.argmin)
        invert_curve(roo1)
        scale_extrema(roo1, maxima)
        invert_curve(roo1)
        scale_extrema(roo1, minima)
    else:
        minima = []
        idx = numpy.argwhere(roo1 > zero).squeeze()
        cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
        b = 0
        for c in cuts:
            minima.append(numpy.argmax(roo1[b:idx[c+1]])+b)
            b = idx[c+1]
        idx = numpy.argwhere(roo1 < zero).squeeze()
        
        roo1 *= -1
        roo1 -= roo1.min()
        roo /= roo.max()
        
        for i in xrange(len(minima)):
            if (i+1) == len(minima):
                b, e = minima[i], len(roo1)
            else:
                b, e = minima[i], minima[i+1]
            val = numpy.max(roo1[b:e])
            roo1[b:e] /=val
        
        roo1 *= -1
        roo1 -= roo1.min()
        roo1 /= roo1.max()
        
        minima = []
        cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
        b = 0
        for c in cuts:
            minima.append(numpy.argmin(roo1[b:idx[c+1]])+b)
            b = idx[c+1]
        
        for i in xrange(len(minima)):
            if (i+1) == len(minima):
                b, e = minima[i], len(roo1)
            else:
                b, e = minima[i], minima[i+1]
            val = numpy.max(roo1[b:e])
            roo1[b:e] /=val
    roo1[0]=0
    
    m = 0 if len(minima) == 0 else minima[0]
    return roo1, m

def normalize(roo1):
    '''
    '''
    
    roo1 -= roo1.min()
    roo1 /= roo1.max()
    return roo1

def invert_curve(roo1):
    '''
    '''
    
    roo1 *= -1
    normalize(roo1)

def find_extrema(roo1, sel, argsel=numpy.argmax):
    '''
    '''
    
    minima = []
    idx = numpy.argwhere(sel).squeeze()
    cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
    if cuts.ndim==0: return minima
    b = 0
    for c in cuts:
        minima.append(argsel(roo1[b:idx[c+1]])+b)
        b = idx[c+1]
    return minima

def scale_extrema(roo1, extrema):
    '''
    '''
    
    for i in xrange(len(extrema)):
        if (i+1) == len(extrema):
            b, e = extrema[i], len(roo1)
        else:
            b, e = extrema[i], extrema[i+1]
        try:
            val = numpy.max(roo1[b:e])
        except:
            _logger.error("%d:%d"%(b,e))
            raise
        if val != 0.0: roo1[b:e] /= val
        
