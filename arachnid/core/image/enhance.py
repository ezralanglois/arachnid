''' Image enhancement routines

.. Created on Jan 13, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy

def histeq(img, hist_bins=256, **extra):
    ''' Equalize the histogram of an image
    
    :Parameters:
    
    img : array
          Image data
    hist_bins : int
                Number of bins for the histogram
    
    :Returns:
    
    img : array
          Histogram equalized image
          
    .. note::
    
        http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    '''
    
    imhist,bins = numpy.histogram(img.flatten(),hist_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = numpy.interp(img.flatten(), bins[:-1], cdf)#use linear interpolation of cdf to find new pixel values
    img = im2.reshape(img.shape)
    return img

def hist_match(img, ref, hist_bins=256, **extra):
    ''' Equalize the histogram of an image
    
    :Parameters:
    
    img : array
          Image data
    hist_bins : int
                Number of bins for the histogram
    
    :Returns:
    
    img : array
          Histogram equalized image
          
    .. note::
    
        http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    '''
    
    imhist,bins = numpy.histogram(ref.flatten(),hist_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    im2 = numpy.interp(img.flatten(), bins[:-1], cdf)#use linear interpolation of cdf to find new pixel values
    img = im2.reshape(img.shape)
    return img

def normalize_standard(img, mask=None, var_one=True, out=None):
    ''' Normalize image to zero mean and one variance
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Mask for mean/std calculation
    var_one : bool
              Set False to disable variance one normalization
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    mdata = img[mask>0.5] if mask is not None else img
    std = numpy.std(mdata) if var_one else 0.0
    out = numpy.subtract(img, numpy.mean(mdata), out)
    if std != 0.0: numpy.divide(out, std, out)
    return out

def normalize_standard_norm(img, mask=None, var_one=True, dust_sigma=2.5, xray_sigma=None, replace=None, out=None):
    ''' Normalize image to zero mean and one variance
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    mask : numpy.ndarray
           Mask for mean/std calculation
    var_one : bool
              Set False to disable variance one normalization
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    mdata = img[mask>0.5] if mask is not None else img
    mdata=replace_outlier(mdata, dust_sigma, xray_sigma, replace, mdata)
    out = numpy.subtract(img, numpy.mean(mdata), out)
    if var_one:
        std = numpy.std(mdata)
        if std != 0.0: numpy.divide(out, std, out)
    return out

def normalize_min_max(img, lower=0.0, upper=1.0, mask=None, out=None):
    ''' Normalize image to given lower and upper range
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    lower : float
            Lower value
    upper : numpy.ndarray
            Upper value
    mask : numpy.ndarray
           Mask for min/max calculation
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    if numpy.issubdtype(img.dtype, numpy.integer):
        img = img.astype(numpy.float)
    
    vmin = numpy.min(img) if mask is None else numpy.min(img*mask)
    out = numpy.subtract(img, vmin, out)
    vmax = numpy.max(img) if mask is None else numpy.max(img*mask)
    if vmax == 0: raise ValueError, "No information in image"
    numpy.divide(out, vmax, out)
    upper = upper-lower
    if upper != 1.0: numpy.multiply(upper, out, out)
    if lower != 0.0: numpy.add(lower, out, out)
    return out

def invert(img, out=None):
    '''Invert an image
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Output image
    '''
        
    out = numpy.multiply(img, -1.0, out)
    normalize_min_max(out, -numpy.max(out), -numpy.min(out), out)
    return out

def vst(img, out=None):
    ''' Variance stablizing transform
    
    :Parameters:

    img : numpy.ndarray
          Input image
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    mval = numpy.min(img)
    if mval < 0: out = numpy.subtract(img, mval, out)
    out = numpy.add(numpy.sqrt(img), numpy.sqrt(img+1), out)
    return out

def replace_outlier(img, dust_sigma, xray_sigma=None, replace=None, out=None):
    '''Clamp outlier pixels, either too black due to dust or too white due to hot-pixels. Replace with 
    samples drawn from the normal distribution with the same mean and standard deviation.
    
    Any random values drawn outside the range of values in the image are clamped to the largest value.
    
    :Parameters:

    img : numpy.ndarray
          Input image
    dust_sigma : float
                 Number of standard deviations for black pixels
    xray_sigma : float
                 Number of standard deviations for white pixels
    replace : float
              Value to replace with, if None use random noise
    out : numpy.ndarray
          Output image
                     
    :Returns:

    out : numpy.ndarray
          Output image
    '''
    
    if out is None: out = img.copy()
    else: out[:]=img
    avg = numpy.mean(img)
    std = numpy.std(img)
    vmin = numpy.min(img)
    vmax = numpy.max(img)
    if vmin == vmax: 
        #print "*****", vmin, '==', vmax
        return out

    
    if xray_sigma is None: xray_sigma=dust_sigma if dust_sigma > 0 else -dust_sigma
    if dust_sigma > 0: dust_sigma = -dust_sigma
    lcut = avg+std*dust_sigma
    hcut = avg+std*xray_sigma
    try:
        vsmin = numpy.min(out[img>=lcut])
    except: 
        vsmin=numpy.min(out)
    try:
        vsmax = numpy.max(out[img<=hcut])
    except: 
        vsmax=numpy.max(out)
    if replace == 'mean':
        replace = numpy.mean(out[numpy.logical_and(out > lcut, out < hcut)])
    if vmin < lcut:
        sel = img < lcut
        if replace is None:
            out[sel] = numpy.random.normal(avg, std, numpy.sum(sel)).astype(out.dtype)
        else: out[sel] = replace
    if vmax > hcut:
        sel = img > hcut
        if replace is None:
            out[sel] = numpy.random.normal(avg, std, numpy.sum(sel)).astype(out.dtype)
        else: out[sel] = replace
    out[img > vsmax]=vsmax
    out[img < vsmin]=vsmin
    return out
