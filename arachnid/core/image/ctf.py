''' CTF utilities including estimate and modeling


.. Created on Dec 4, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, scipy, scipy.fftpack, math
import scipy.misc, scipy.linalg, scipy.special
import ndimage_utility, eman2_utility
import heapq

from ..app import tracing
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
DEBUG=1

try: 
    from spi import _spider_ctf
    _spider_ctf;
except:
    _spider_ctf=None
    tracing.log_import_error('Failed to load _spider_ctf.so module', _logger)

try: 
    from util import _ctf
    _ctf;
except:
    _ctf=None
    tracing.log_import_error('Failed to load _ctf.so module', _logger)

#Segment power spectra
# Fit defocus
# Fit ellipse
# 
#http://nicky.vanforeest.com/_downloads/fitEllipse.py

######################################################################
# Fast Def
######################################################################


def normalize_wb(img, rmin, rmax, sel):
    '''
    '''
    
    assert(numpy.sum(sel)>0)
    img_mod = img.copy()
    rang = (rmax-rmin)/2.0
    freq2 = float(img.shape[0])/(rang/1);
    freq1 = float(img.shape[0])/(rang/15);
    img_norm = ndimage_utility.filter_annular_bp(img, freq1, freq2)
    sph = ndimage_utility.spiral_transform(img_norm)
    
    nsel = numpy.logical_not(sel)
    img_norm[nsel]=0
    img_mod[nsel]=0
    
    '''
    img_mod[sel] = numpy.sqrt( numpy.power(numpy.abs(sph[sel]), 2.0) + numpy.power(img_norm[sel], 2.0) )
    img_norm[sel] = numpy.cos( numpy.arctan2(numpy.abs(sph[sel]), img_norm[sel]) )
    '''
    for i in xrange(img_norm.shape[0]):
        for j in xrange(img_norm.shape[1]):
            if not sel[i, j]: continue
            tmp = numpy.abs(sph[i, j])
            img_mod[i, j] = numpy.sqrt( numpy.power(tmp, 2.0) + numpy.power(img_norm[i, j], 2))
            img_norm[i, j] = numpy.cos( numpy.arctan2(tmp, img_norm[i, j]))
    return img_norm, img_mod


def wrapped_phase(sph, dir, img_norm, roi):
    '''
    '''
    
    wphase = numpy.zeros(img_norm.shape)
    ormod = numpy.zeros(img_norm.shape)
    cen = int( float(roi.shape[0])/2.0 )
    ci = numpy.complex(0, 1.0)
    cut = 0.1
    for i in xrange(img_norm.shape[0]):
        for j in xrange(img_norm.shape[1]):
            if roi[i,j]:
                tmp = -ci*numpy.exp(ci*dir[i,j])*sph[i,j]
                wphase[i,j] = numpy.arctan2(tmp.real, img_norm[i,j])
                ormod[i,j] = numpy.sqrt( tmp.real*tmp.real + img_norm[i,j]*img_norm[i,j] )
                tmp_theta = numpy.arctan2(j-cen, i-cen)
                if not ( ( (tmp_theta>cut) | (tmp_theta<-cut) )  and not (tmp_theta>numpy.pi-cut or tmp_theta<-numpy.pi+cut)  and not ( not(tmp_theta>numpy.pi/2+cut) and tmp_theta>numpy.pi/2-cut) and not ( not(tmp_theta>-numpy.pi/2+cut) and  tmp_theta > -numpy.pi-cut ) ):
                    roi[i,j]=0
            else:ormod[i,j]=0
            
    return wphase, ormod

def unwrapping(wphase, ormod, lam, size):
    '''
    '''
    
    unwphase = numpy.zeros(wphase.shape, dtype=wphase.dtype)
    max_levels=10
    min_quality=0.01
    numpy.abs(ormod, ormod)
    jmax, imax = numpy.unravel_index(ormod.argmax(), ormod.shape)
    quality_vals = ((ormod/ormod[imax,jmax])*(max_levels-1)).round()
    
    # pre-compute
    gaussian = numpy.zeros((2*size+1, 2*size+1))
    var = 1.5
    center = float(gaussian.shape[0])/2.0
    for i in xrange(0, gaussian.shape[0]):
        for j in xrange(0, gaussian.shape[1]):
            gaussian[i,j] = numpy.exp(-( (i-center)**2+(j-center)**2 ) / (2*var*var))
    
    processed = numpy.zeros(wphase.shape, dtype=numpy.bool)
    
    ind_i = numpy.zeros(8, dtype=numpy.int)
    ind_j = numpy.zeros(8, dtype=numpy.int)
    unwphase[imax, jmax]=wphase[imax, jmax]
    processed[imax, jmax]=True
    pqueue = []
    heapq.heappush(pqueue, (-quality_vals[imax, jmax], imax, jmax)) # pos or neg?
    pred, cor, norm = 0.0, 0.0, 0.0
    while len(pqueue)>0:
        i, j = heapq.heappop(pqueue)[1:]
        ind_i[:] = [i-1, i-1, i-1, i,   i  , i+1, i+1, i+1]
        ind_j[:] = [j-1, j,   j+1, j-1, j+1, j-1, j,   j+1]
        for k in xrange(8):
            ni, nj = ind_i[k], ind_j[k]
            if not processed[ni, nj] and \
            ormod[ni,nj]> min_quality and \
            (ni-size)>0 and (nj-size)>0 \
            and (ni+size)<(processed.shape[0]-1) \
            and (nj+size)<(processed.shape[1]-1):
                wp = wphase[ni,nj]
                for li in xrange(-size, size+1):
                    nli=ni+li
                    for lj in xrange(-size, size+1):
                        nlj=nj+lj
                        if processed[nli, nlj]:
                            uw = unwphase[nli, nlj]
                            q = ormod[nli, nlj]
                            g = gaussian[li+size, lj+size]
                            t = wp - uw
                            n = (numpy.floor(t+numpy.pi)/(2*numpy.pi)) if t > 0 else (numpy.ceil(t-numpy.pi)/(2*numpy.pi))
                            up = t - (2*numpy.pi)*n
                            pred += uw*q*g
                            cor += up*q*g
                            norm += q*g
                unwphase[ni, nj] = pred/norm + (lam*cor)/norm
                processed[ni, nj]=True
                norm, pred, cor = 0.0, 0.0, 0.0
                heapq.heappush(pqueue, (-quality_vals[ni, nj], ni, nj))
    return unwphase

def zernike_order(n):
    '''
    '''
    
    return numpy.ceil((-3+numpy.sqrt(9+8*float(n)))/2.)

def fit_zernikes(coefs, phase, mod2, roi):
    '''http://wiki.scipy.org/Cookbook/FittingData#head-5eba0779a34c07f5a596bbcf99dbc7886eac18e5
    Code adopted from: http://xmipp.cnb.csic.es/~xmipp/trunk/xmipp/documentation/html/xmipp__polynomials_8cpp_source.html
    '''
    
    # set zero to center!
    
    coefs = coefs.astype(numpy.int)
    num_zer = int(coefs.sum())
    porder = int(zernike_order(len(coefs)))
    imax_dim2 =  2.0/max(phase.shape[0],phase.shape[1])
    pval = numpy.zeros((porder,porder))
    
    assert(roi.sum()>0)
    A = numpy.zeros((int(roi.sum()), num_zer))
    b = numpy.zeros(A.shape[0])
    w = numpy.zeros(b.shape)
    
    # Zernike Polynomials
    # Efficient Cartesian representation of Zernike polynomials in computer memory SPIE Vol. 3190 pp. 382
    # pre-compute
    binom = scipy.special.binom
    fact = scipy.misc.factorial
    fmat_list = []
    for nz in xrange(coefs.shape[0]):
        if coefs[nz] != 0:
            n = int(zernike_order(nz))
            fmat = numpy.zeros((n+1,n+1), dtype=numpy.int)
            l = 2*nz-n*(n+2)
            p = l>0
            q = numpy.abs(l)-1/2 if (numpy.mod(n,2)!=0) else numpy.abs(l)/2-1 if l>0 else numpy.abs(l)/2
            l = numpy.abs(l)
            m = (n-l)/2
            for i in xrange(q+1):
                k1 = binom(l,2*i+p)
                for j in xrange(m+1):
                    factor = 1.0 if numpy.mod((i+j), 2) == 0 else -1
                    k2 = factor*k1*fact(n-j)/fact(j)*fact(m-j)*fact(n-m-j)
                    for k in xrange(m-j+1):
                        ypow = 2*(i+k)+p
                        xpow = n-2*(i+j+k)-p
                        fmat[xpow, ypow] += k2*binom(m-j,k)

        else:
            fmat = numpy.zeros((1,1), dtype=numpy.int)
            fmat[0,0]=0
        fmat_list.append(fmat)
    
    pixel_idx = 0
    cent = phase.shape[0]/2
    #Prepare matrix vector and weights
    for i in xrange(phase.shape[0]):
        ic = i-cent
        for j in xrange(phase.shape[1]):
            if not roi[i,j]: continue
            jc = j-cent
            y = ic*imax_dim2
            x = jc*imax_dim2
            
            for py in xrange(porder):
                ypy = numpy.power(y, py)
                for px in xrange(porder):
                    pval[px, py] = ypy*numpy.power(x, px)
            
            for k in xrange(num_zer):
                tmp = 0.0
                fmat = fmat_list[k]
                for px in xrange(fmat.shape[0]):
                    for py in xrange(fmat.shape[1]):
                        tmp += fmat[py,px]*pval[py,px]
                A[pixel_idx, k] = tmp
            b[pixel_idx] = phase[i,j]
            w[pixel_idx] = numpy.abs(mod2[i,j])
            pixel_idx+=1
    
    # Apply weights
    assert(w.shape[0]==A.shape[0])
    for i in xrange(w.shape[0]):
        wii = numpy.sqrt(w[i])
        b[i]*=wii
        for j in xrange(A.shape[1]):
            A[i,j]*=wii
    
    # Fit least squares
    zern_coef = scipy.linalg.lstsq(A, b)[0]
    
    
    # update roi   
    pixel_idx=0 
    for i in xrange(phase.shape[0]):
        for j in xrange(phase.shape[1]):
            if not roi[i,j]: continue
            tmp=0.0
            for k in xrange(num_zer):
                tmp += A[pixel_idx, k]*zern_coef[k]
            if (numpy.fabs(tmp)-phase[i,j]) > numpy.pi:
                roi[i,j]=False
            pixel_idx+=1

    return zern_coef
    
def demodulate(pow, lam, size, rmin, rmax):#0.74688*4
    '''
    '''
    
    coefs = numpy.zeros(13)
    coefs[numpy.asarray((0,3,4,5,12))]=1
    
    roi = ndimage_utility.model_ring(rmin, rmax, pow.shape, dtype=numpy.bool)
    x,y = ndimage_utility.grid_image(pow.shape)
    dir = -numpy.arctan2(x, y)
    img_norm, img_mod = normalize_wb(pow, rmin, rmax, roi)
    
    
    img_mod /= img_mod.max()
    sph = ndimage_utility.spiral_transform(img_norm)
    wphase, ormod = wrapped_phase(sph, dir, img_norm, roi)
    phase = unwrapping(wphase, ormod, lam, size)
    
    if DEBUG:
        import ndimage_file
        ndimage_file.write_image('pow.spi', pow)
        ndimage_file.write_image('pow_norm.spi', img_norm)
        ndimage_file.write_image('wphase.spi', wphase)
        ndimage_file.write_image('phase.spi', phase)
    
    mod2 = numpy.ones(phase.shape)
    zern_coef = fit_zernikes(coefs, phase, mod2, roi)
    if roi.sum() > 0:
        zern_coef = fit_zernikes(coefs, phase, mod2, roi)
    else: print "Skipped second fitting"
    
    index=0
    for i in xrange(coefs.shape[0]):
        if coefs[i] != 0:
            coefs[i] = zern_coef[index]
            index+=1
    return coefs

def ctf_model_zernike(coefs, apix, voltage, **extra):
    '''
    '''
    
    Z8=coefs[4]
    Z3=coefs[12]
    Z4=coefs[3]
    Z5=coefs[5]
    voltage*= 1000
    lambd = 12.2643247/numpy.sqrt(voltage*(1.+0.978466e-6*voltage));
    return numpy.fabs(2*apix*apix*(2*Z3-6*Z8)/(numpy.pi*lambd)), \
           numpy.fabs(2*apix*apix*(numpy.sqrt(Z4*Z4+Z5*Z5))/(numpy.pi*lambd)), \
           0.5*numpy.rad2deg(numpy.arctan2(Z5,Z4))+90.0
    
def estimate_defocus_fast(pow, rmin, rmax, eps_phase=2.0, window_phase=10, **extra):
    '''
    '''
    
    if rmin < rmax: rmin, rmax = rmax, rmin
    min_freq = extra['apix']/rmin
    max_freq = extra['apix']/rmax
    models = []
    test = numpy.zeros(4)
    for scale in (1, 0.76, 0.88, 0.64):
        coefs = demodulate(pow, eps_phase, window_phase, scale*min_freq*pow.shape[0], scale*max_freq*pow.shape[0])
        defocus, asig_mag, asig_ang = ctf_model_zernike(coefs, **extra)
        print scale, ' - ', defocus, asig_mag, asig_ang
        models.append( (defocus, asig_mag, asig_ang) )
    test = numpy.asarray([d[0] for d in models])
    test -= test.mean()
    numpy.fabs(test, test)
    defocus, asig_mag, asig_ang = models[test.argmin()]
    
    defocus_u=defocus+asig_mag
    defocus_v=defocus-asig_mag
    return defocus_u, defocus_v, numpy.deg2rad(asig_ang)
        

def correct(img, ctfimg):
    ''' Corret the CTF of an image
    
    :Parameters:
    
    :Returns:
    
    '''
    
    '''
        fimg = scipy.fftpack.fftn(img)
        #fimg = numpy.fft.fftshift(fimg)
        fimg *= ctfimg.T
        return scipy.fftpack.ifftn(img).real
    '''
    nsam = (img.shape[1]+2) if (img.shape[1]%2) == 0 else (img.shape[1]+1)
    out = numpy.zeros((img.shape[0], nsam), dtype=img.dtype)
    out[:, :img.shape[1]] = img
    _spider_ctf.correct_image(out.T, ctfimg.T, out.shape[0])
    return out[:, :img.shape[1]]

def phase_flip_transfer_function(out, defocus, cs, ampcont, envelope_half_width=10000, voltage=None, elambda=None, apix=None, maximum_spatial_freq=None, source=0.0, defocus_spread=0.0, astigmatism=0.0, azimuth=0.0, ctf_sign=-1.0, **extra):
    ''' Create a transfer function for phase flipping
    
    :Parameters:
    
    out : size, tuple, array
          Size of square ransfer function image, tuple of dimensions, or image
    defocus : float
              Amount of defocus, in Angstroems
    cs : object
         Spherical aberration constant
    ampcont : float
              Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
    envelope_half_width : float
                          Envelope parameter specifies the 2 sigma level of the Gaussian
    voltage : float
              Voltage of microscope (Defaut: None)
    elambda : float
              Wavelength of the electrons (Defaut: None)
    apix : float
           Size of pixel in angstroms  (Defaut: None)
    maximum_spatial_freq : float
                           Spatial frequency radius corresponding to the maximum radius (Defaut: None)
    source : float
             Size of the illumination source in reciprocal Angstroems
    defocus_spread : float
                     Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
    astigmatism : float
                  Defocus difference due to axial astigmatism (Defaut: 0)
    azimuth : float
              Angle, in degrees, that characterizes the direction of astigmatism (Defaut: 0)
    ctf_sign : float
               Application of the transfer function results in contrast reversal if underfocus (Defaut: -1)
    
    :Returns:
    
    out : array
          Transfer function image
    '''
    
    if elambda is None: 
        if voltage is None: raise ValueError, "Wavelength of the electrons is not set as elambda or voltage"
        elambda = 12.398 / math.sqrt(voltage * (1022.0 + voltage))
    if maximum_spatial_freq is None: 
        if apix is None: raise ValueError, "Patial frequency radius corresponding to the maximum radius is not set as maximum_spatial_freq or apix"
        maximum_spatial_freq = 0.5/apix
        
    if isinstance(out, tuple):
        nsam = (out[0]+2)/2 if (out[0]%2) == 0 else (out[0]+1)/2
        out = numpy.zeros((out[1], nsam), dtype=numpy.complex64)
    elif isinstance(out, int):
        nsam = (out+2)/2 if (out%2) == 0 else (out+1)/2
        out = numpy.zeros((out, nsam), dtype=numpy.complex64)
    else:
        if out.dtype != numpy.complex64: raise ValueError, "Requires complex64 for out"
    _spider_ctf.transfer_function_phase_flip_2d(out.T, out.shape[0], float(cs), float(defocus), float(maximum_spatial_freq), float(elambda), float(source), float(defocus_spread), float(astigmatism), float(azimuth), float(ampcont), 0.0, int(ctf_sign))
    return out

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

def energy_cutoff(roo, energy=0.95):
    ''' Determine the index when the energy drops below cutoff
    
    :Parameters:
    
    roo : array
          1-D rotatational average of the power spectra
    energy : float
             Energy cutoff
    
    :Returns:
    
    offset : int
             Offset when energy drops below cutoff
    '''
    
    return numpy.searchsorted(numpy.cumsum(roo)/roo.sum(), energy)

def noise_model(p, freq):
    '''Model describing the noise growth
    
    :Parameters:
    
    p : array
        Array of parameters
    freq : array
          Array of frequencey values
    
    :Returns:
    
    model : array
            Estimated model
    '''
    
    return p[0] + p[1] * numpy.sqrt(freq) + p[2]*freq + p[3]*numpy.square(freq)

def bfactor_model(p, freq):
    '''Expoential model describing the background factor decay
    
    :Parameters:
    
    p : array
        Array of parameters
    freq : array
          Array of frequencey values
    
    :Returns:
    
    model : array
            Estimated model
    '''
    
    return p[0] + p[1] * numpy.sqrt(freq) + p[2]*freq + p[3]*numpy.square(freq)

def refine_defocus(roo, sfreq, defocus, n, defocus_start=1.2, defocus_end=7.0, **extra):
    '''Estimate the defocus of a background subtract power spectrum
    
    :Parameters:
    
    roo : array
          Background subtracted power spectrum
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    defocus_start : float
                    Start of the defocus range to search
    defocus_end : float
                  End of the defocus range to search
    
    :Returns:
    
    defocus : float
              Defocus (A)
    err : float
          Error between model and powerspectra
    '''
    
    '''
    http://stackoverflow.com/questions/13405053/scipy-leastsq-fit-to-a-sine-wave-failing
    yhat = fftpack.rfft(yReal)
    idx = (yhat**2).argmax()
    freqs = fftpack.rfftfreq(N, d = (xReal[1]-xReal[0])/(2*pi))
    frequency = freqs[idx]
    '''
    
    def errfunc(p, y, sfreq, n, ampcont, voltage):
        return y-ctf_model_full(p[0], sfreq, n, ampcont, p[1], voltage, p[2])**2

    
    sfreq4, sfreq2, w, factor = ctf_model_precalc(sfreq, n, **extra)
    p0 = [defocus, extra['cs'], extra['apix']]
    defocus = scipy.optimize.leastsq(errfunc,p0,args=(roo, sfreq, n, extra['ampcont'], extra['voltage']))[0]
    ecurve = errfunc(defocus, roo, sfreq, n, extra['ampcont'], extra['voltage'])
    err = numpy.sqrt(numpy.sum(numpy.square(ecurve)))

    
    defocus[0] *= factor*1e4
    return defocus, err, ecurve

def first_zero(roo, **extra):
    ''' Determine the first zero of the CTF
    '''
    
    if 1 == 1:
        roo = roo[2:]
        zero = numpy.mean(roo[len(roo)-len(roo)/5:])
        minima = []
        idx = numpy.argwhere(roo < zero).squeeze()
        cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
        b = 0
        for c in cuts:
            minima.append(numpy.argmin(roo[b:idx[c+1]])+b)
            b = idx[c+1]
        #n = len(minima)-1 if len(minima) < 2 else 1
        if len(minima)==0:return 0
        return minima[0]+2
         
    
    sfreq = numpy.arange(len(roo), dtype=numpy.float)
    defocus = estimate_defocus_spi(roo, sfreq, len(roo), **extra)[0]
    model = ctf_model_spi(defocus, sfreq, len(roo), **extra)**2
    model -= model.min()
    model /= model.max()
    idx = numpy.argwhere(model > 0.5).squeeze()
    cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
    try:
        return numpy.argmax(roo[:idx[cuts[2]+1]])
    except:
        try:
            return numpy.argmax(roo[:idx[cuts[1]+1]])
        except:
            return numpy.argmax(roo[:idx[cuts[0]+1]]) if len(cuts) > 0 else 0

def resolution(e, n, apix, **extra):
    ''' Estimate resolution of frequency
    '''
    
    return apix/( (0.5*e)/float(n) )

def estimate_defocus_spi(roo, sfreq, n, defocus_start=0.2, defocus_end=7.0, **extra):
    '''Estimate the defocus of a background subtract power spectrum
    
    :Parameters:
    
    roo : array
          Background subtracted power spectrum
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    defocus_start : float
                    Start of the defocus range to search
    defocus_end : float
                  End of the defocus range to search
    
    :Returns:
    
    defocus : float
              Defocus (A)
    err : float
          Error between model and powerspectra
    '''
    
    def errfunc(p, y, sfreq, n, ampcont, cs, voltage, apix):
        return y-ctf_model_spi(p[0], sfreq, n, ampcont, cs, voltage, apix)**2
    
    defocus = None
    err = 1e20
    #roo = roo**2
    
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        p0 = [p*1e4]
        try:
            dz1, = scipy.optimize.leastsq(errfunc,p0,args=(roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix']))[0]
        except: continue
        err1 = numpy.sqrt(numpy.sum(numpy.square(errfunc([dz1], roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix']))))
        if err1 < err and dz1 > 0:
            err = err1
            defocus=dz1
    if defocus is None: return 0, 0, 0
    return defocus, err, errfunc([defocus], roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix'])

def estimate_defocus_orig(roo, sfreq, n, defocus_start=0.2, defocus_end=7.0, **extra):
    '''Estimate the defocus of a background subtract power spectrum
    
    :Parameters:
    
    roo : array
          Background subtracted power spectrum
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    defocus_start : float
                    Start of the defocus range to search
    defocus_end : float
                  End of the defocus range to search
    
    :Returns:
    
    defocus : float
              Defocus (A)
    err : float
          Error between model and powerspectra
    '''
    
    def errfunc(p, y, sfreq, n, ampcont, cs, voltage, apix):
        return y-ctf_model_orig(p[0], sfreq, n, ampcont, cs, voltage, apix)**2
    
    defocus = None
    err = 1e20
    #roo = roo**2
    
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        p0 = [p*1e4]
        dz1, = scipy.optimize.leastsq(errfunc,p0,args=(roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix']))[0]
        err1 = numpy.sqrt(numpy.sum(numpy.square(errfunc([dz1], roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix']))))
        if err1 < err:
            err = err1
            defocus=dz1
    
    return defocus, err, errfunc([defocus], roo, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix'])

def estimate_background(raw, defocus, sfreq, n, defocus_start=0.2, defocus_end=7.0, **extra):
    '''Estimate the defocus of a background subtract power spectrum
    
    :Parameters:
    
    roo : array
          Background subtracted power spectrum
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    defocus_start : float
                    Start of the defocus range to search
    defocus_end : float
                  End of the defocus range to search
    
    :Returns:
    
    defocus : float
              Defocus (A)
    err : float
          Error between model and powerspectra
    '''
    
    def errfunc(p, y, sfreq, n, ampcont, cs, voltage, apix):
        return subtract_background(y, p[0])-ctf_model_orig(defocus, sfreq, n, ampcont, cs, voltage, apix)**2
    
    bg = None
    err = 1e20
    #roo = roo**2
    
    for bg1 in xrange(3,90,2):
        err1 = numpy.sqrt(numpy.sum(numpy.square(errfunc([bg1], raw, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix']))))
        if err1 < err:
            err = err1
            bg=bg1
    
    return bg, err, errfunc([bg], raw, sfreq, n, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix'])

def estimate_defocus(roo, sfreq, n, defocus_start=1.2, defocus_end=7.0, **extra):
    '''Estimate the defocus of a background subtract power spectrum
    
    :Parameters:
    
    roo : array
          Background subtracted power spectrum
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    defocus_start : float
                    Start of the defocus range to search
    defocus_end : float
                  End of the defocus range to search
    
    :Returns:
    
    defocus : float
              Defocus (A)
    err : float
          Error between model and powerspectra
    '''
    
    '''
    http://stackoverflow.com/questions/13405053/scipy-leastsq-fit-to-a-sine-wave-failing
    yhat = fftpack.rfft(yReal)
    idx = (yhat**2).argmax()
    freqs = fftpack.rfftfreq(N, d = (xReal[1]-xReal[0])/(2*pi))
    frequency = freqs[idx]
    '''
    
    def errfunc(p, y, sfreq4, sfreq2, w):
        return y-ctf_model_calc(p[0], sfreq4, sfreq2, w)**2
    
    defocus = None
    err = 1e20
    #roo = roo**2
    
    sfreq4, sfreq2, w, factor = ctf_model_precalc(sfreq, n, **extra)
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        p0 = [p/factor]
        dz1, = scipy.optimize.leastsq(errfunc,p0,args=(roo, sfreq4, sfreq2, w))[0]
        err1 = numpy.sqrt(numpy.sum(numpy.square(errfunc([dz1], roo, sfreq4, sfreq2, w))))
        if err1 < err:
            err = err1
            defocus=dz1
    
    if 1 == 0:
        p0=[initial_guess(roo, sfreq, n, **extra)/1e4]
        print "guess: ", p0[0]*factor*1e4
        dz2, = scipy.optimize.leastsq(errfunc,p0,args=(roo, sfreq4, sfreq2, w))[0]
        err2 = numpy.sqrt(numpy.sum(numpy.square(errfunc([dz1], roo, sfreq4, sfreq2, w))))
        print "Defocus: ", dz1*factor*1e4, dz2*factor*1e4
        print "Error: ", err, err2
    
    return defocus*factor*1e4, err, errfunc([defocus], roo, sfreq4, sfreq2, w)

def background(roo, window):
    ''' Subtract the background using a moving average filter
    
    :Parameters:
    
    roo : array
          Power spectra 1D
    window : int
             Size of the window
    
    :Returns:
    
    out : array
          Background subtracted power spectra
    '''
    
    bg = roo.copy()
    off = int(window/2.0)
    weightings = numpy.ones(window)
    weightings /= weightings.sum()
    bg[off:len(roo)-off]=numpy.convolve(roo, weightings)[window-1:-(window-1)]
    return bg

def subtract_background(roo, window):
    ''' Subtract the background using a moving average filter
    
    :Parameters:
    
    roo : array
          Power spectra 1D
    window : int
             Size of the window
    
    :Returns:
    
    out : array
          Background subtracted power spectra
    '''
    
    bg = roo.copy()
    off = int(window/2.0)
    weightings = numpy.ones(window)
    weightings /= weightings.sum()
    bg[off:len(roo)-off]=numpy.convolve(roo, weightings)[window-1:-(window-1)]
    return roo-bg

def ctf_model_calc(dz1, sfreq4, sfreq2, w):
    ''' Build a CTF model curve from precalculated values
    
    :Parameters:
    
    dz1 : float
          Defocus scaled by CS and the voltage in (nm)
    sfreq4 : array
             Frequency array accounting for CS and voltage to the 4th power and scaled by 1/2 pi
    sfreq2 : array
             Frequency array accounting for CS and voltage to the 2d power and scaled by pi
    w : float
        Offset based on amplitude contrast
             
    :Returns:
    
    out : array
          CTF model
    '''
    
    return numpy.sin( sfreq4 - dz1*sfreq2*1e4 + w )

def ctf_model_precalc(sfreq, n, ampcont, cs, voltage, apix, **extra):
    ''' Precalculate arrays and values for the CTF model
    
    :Parameters:
    
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
    extra : dict
            Unused keyword arguments
             
    :Returns:

    sfreq4 : array
             Frequency array accounting for CS and voltage to the 4th power and scaled by 1/2 pi
    sfreq2 : array
             Frequency array accounting for CS and voltage to the 2d power and scaled by pi
    w : float
        Offset based on amplitude contrast
    factor : float
             Convert the defocus value to the proper units
    '''
    
    sfreq = sfreq.astype(numpy.float)
    cs *= 1e7
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    w = numpy.arcsin(ampcont)
    factor = numpy.sqrt(cs*lam)
    
    km1 = numpy.sqrt(numpy.sqrt(cs*lam**3)) / (2.0*apix)
    dk = km1/n
    sfreq = sfreq*dk

    sfreq4 = 0.5*numpy.pi*sfreq**4
    sfreq2 = numpy.pi*sfreq**2
    return sfreq4, sfreq2, w, factor

def frequency(sfreq, n, apix, **extra):
    '''
    '''
    
    k = 1.0/(2.0*apix)
    dk = k/n
    return sfreq.astype(numpy.float)*dk

def model_1D_to_2D(model, out=None):
    '''
    '''
    
    if out is None: out = numpy.zeros((len(model)*2, len(model)*2))
    out[len(model), len(model):] = model
    return ndimage_utility.rotavg(out)



def ctf_model_orig(dz1, sfreq, n, ampcont, cs, voltage, apix, **extra):
    ''' CTF model with non-generalized variables dependent on voltage
    
    Useful for CS-correct microscopes when the CS is 0
    
    :Parameters:
    
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
             
    :Returns:

    ctf : array
          CTF 1D
    '''
    
    sfreq = sfreq.astype(numpy.float)
    cs *= 1e7
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    w = numpy.arcsin(ampcont)
    
    km1 = 1.0 / (2.0*apix)
    dk = km1/n
    sfreq = sfreq*dk
    
    sfreq4 = lam**3*cs*0.5*numpy.pi*sfreq**4
    sfreq2 = lam*numpy.pi*sfreq**2
    
    return numpy.sin( sfreq4 - dz1*sfreq2 + w) #*1e4 + w )


######################################################################
# CTF FIND3
######################################################################

def search_model_2d(pow, dfmin, dfmax, fstep, rmin, rmax, ampcont, cs, voltage, pad=1.0, apix=None, xmag=None, res=None, pre_decimate=0, **extra):
    '''
    '''
    
    if rmin < rmax: rmin, rmax = rmax, rmin
    if dfmax < dfmin: dfmin, dfmax = dfmax, dfmin
    if apix is None: apix = res*(10.0**4.0)/xmag
    if rmin > 50: rmin = 50.0
    if rmax > rmin: raise ValueError, "Increase rmax, rmin cannot be greater than 50"
    apix *= pad
    rmin = apix/rmin
    rmax = apix/rmax
    rmin2, rmax2 = rmin**2,rmax**2
    hw = -1.0/rmax2
    print 'Searching CTF Parameters...'
    print '      DFMID1      DFMID2      ANGAST          CC'
    best=(-1e20, None, None, None)
    i1 = int(dfmin/fstep)
    i2 = int(dfmax/fstep)
    cs *= 10**7.0
    kv = voltage*1000.0
    wl = 12.26/numpy.sqrt( kv+0.9785*kv**2/10.0**6.0 )
    
    pow = pow.astype(numpy.float32)
    if pre_decimate > 0:
        pow_sm = eman2_utility.decimate(pow, float(pow.shape[0])/pre_decimate)
    else: pow_sm = pow
    print pre_decimate, pow_sm.shape, pow.shape
    #pow = pow.T.copy()
    pow_sm = pow_sm[:, :pow_sm.shape[0]/2].copy()
    thetatr = wl/(apix*min(pow_sm.shape))
    smooth_2d(pow_sm, rmin)
                    
    p=0
    for k in xrange(0, 18):
        for i in xrange(i1, i2+1):
            for j in xrange(i1, i2+1):
                df1, df2, ang = fstep*i, fstep*j, numpy.deg2rad(5.0*k)
                val = eval_model_2d(df1, df2, ang, pow_sm, cs, wl, ampcont, thetatr, rmin2, rmax2, hw)
                #model[:, :]=0
                #val = eval_model_2d_test(df1, df2, ang, pow, model, cs, wl, ampcont, thetatr, rmin2, rmax2, hw)
                if val > best[0]:  
                    print "%f\t%f\t%f\t%f"%(df1, df2, numpy.rad2deg(ang), val)
                    best=(val, df1, df2, ang)
                p+=1
                
    if pre_decimate > 0 and 1 == 0:
        smooth_2d(pow, rmin)
        thetatr = wl/(apix*min(pow.shape))
        pow = pow[:, :pow.shape[0]/2].copy()
    else:
        pow = pow_sm
    
    def error_func(p0):
        '''
        '''
        return -eval_model_2d(p0[0], p0[1], p0[2], pow, cs, wl, ampcont, thetatr, rmin2, rmax2, hw)
    
    vals = scipy.optimize.fmin(error_func, (best[1], best[2], best[3]), disp=1)
    print vals
    return vals[0], vals[1], vals[2]
    
def smooth_2d(pow, rmin):
    '''
    '''
    
    n = max(pow.shape)
    buf = numpy.zeros((n,n), dtype=numpy.float32)
    nw = int(pow.shape[0]*rmin*numpy.sqrt(2.0))
    size = numpy.asarray(pow.shape, dtype=numpy.int32)
    _ctf.msmooth(pow.T, size, int(nw), buf.T)

def ctf_2d(pow, dfmid1, dfmid2, angast, ampcont, cs, voltage, apix=None, xmag=None, res=None, **extra):
    '''
          DO 200 L=1,JXYZ(1)/2
        LL=L-1
        DO 200 M=1,JXYZ(2)
          MM=M-1
          IF (MM.GT.JXYZ(2)/2) MM=MM-JXYZ(2)
          ID=L+JXYZ(1)/2*(M-1)
            I=L+JXYZ(1)/2
            J=M+JXYZ(2)/2
            IF (J.GT.JXYZ(2)) J=J-JXYZ(2)
            IS=I+JXYZ(1)*(J-1)
C            OUT(IS)=POWER(ID)/DRMS1*SQRT(2.0*PI)
            OUT(IS)=POWER(ID)/DRMS1/2.0+0.5
            IF (OUT(IS).GT.1.0) OUT(IS)=1.0
            IF (OUT(IS).LT.-1.0) OUT(IS)=-1.0
          RES2=(REAL(LL)/JXYZ(1))**2+(REAL(MM)/JXYZ(2))**2
          IF ((RES2.LE.RMAX2).AND.(RES2.GE.RMIN2)) THEN
            CTFV=CTF(CS,WL,WGH1,WGH2,DFMID1,DFMID2,
     +               ANGAST,THETATR,LL,MM)
             I=JXYZ(1)/2-L+1
              J=JXYZ(2)-J+2
              IF (J.LE.JXYZ(2)) THEN
                IS=I+JXYZ(1)*(J-1)
              OUT(IS)=CTFV**2
              ENDIF
          ENDIF
    '''
    
    if apix is None: apix = res*(10.0**4.0)/xmag
    cs *= 10**7.0
    kv = voltage*1000.0
    wl = 12.26/numpy.sqrt( kv+0.9785*kv**2/10.0**6.0 )
    thetatr = wl/(apix*pow.shape[1])
    for l in xrange(pow.shape[1]/2):
        for m in xrange(pow.shape[0]):
            if m > pow.shape[1]/2: m = m - pow.shape[1] + 1
            i = pow.shape[0]/2-l
            j = m + pow.shape[1]/2
            if j > pow.shape[1]: j = j - pow.shape[1] + 1
            j=pow.shape[1]-j+1
            if j>=pow.shape[1]: print j, pow.shape[1]
            if i>=pow.shape[0]: print i, pow.shape[0]
            assert(j<pow.shape[1])
            assert(i<pow.shape[0])
            pow.ravel()[j+pow.shape[1]*i] = ctf_2d_value(l, m, dfmid1, dfmid2, angast, cs, wl, ampcont, thetatr)**2

def ctf_2d_value(ix, iy, dfmid1, dfmid2, angast, cs, wl, wgh, thetatr):
    '''
    '''
    
    return _ctf.ctf(float(cs), float(wl), float(numpy.sqrt(1.0-wgh*wgh)), float(wgh), float(dfmid1), float(dfmid2), float(angast), float(thetatr), int(ix), int(iy))


def eval_model_2d(dfmid1, dfmid2, angast, pow, cs, wl, wgh, thetatr, rmin2, rmax2, hw):
    '''
    '''
    
    n = max(pow.shape)
    return _ctf.evalctf(float(cs), float(wl), float(numpy.sqrt(1.0-wgh*wgh)), float(wgh), float(dfmid1), float(dfmid2), float(angast), float(thetatr), float(hw), pow.T, numpy.asarray((n, n, 0), dtype=numpy.int32), float(rmin2), float(rmax2), 0.0)

def eval_model_2d_test(dfmid1, dfmid2, angast, pow, model, cs, wl, wgh, thetatr, rmin2, rmax2, hw):
    '''
    '''
    
    n = max(pow.shape)
    return _ctf.evalctf_test(float(cs), float(wl), float(numpy.sqrt(1.0-wgh*wgh)), float(wgh), float(dfmid1), float(dfmid2), float(angast), float(thetatr), float(hw), pow.T, model.T, numpy.asarray((n, n, 0), dtype=numpy.int32), float(rmin2), float(rmax2), 0.0)

"""
def assess_model_2d(pow, ampcont, cs, voltage, apix, angast, rmin2, rmax2, wgh1, wgh2, thetatr, wl, hw):#, dast=0.0):
    ''' book flight oct 23, 29
    Adapted from CTFFIND3
    '''
    
    ddif = dfmid1 - dfmid2
    dsum = dfmid1 + dfmid2
    half_thetatrsq = 0.5*thetatr*thetatr
    twopi_wli  = 2.0*numpy.pi/wl 
    cnt = 0
    sum, sum1, sum2 = 0, 0, 0
    for m in xrange(pow.shape[1]):
        if m > pow.shape[1]/2: m = m - pow.shape[1]
        realpart2 = float(m) / pow.shape[1]
        for l in xrange(pow.shape[0]):
            realpart1 = float(l) / pow.shape[0] 
            res2 = realpart1*realpart1 + realpart2*realpart2
            if res2 > rmax2 or res2 <= rmin2: continue
            rad2 = l*l + m*m
            if rad2 > 0.0:
                hang2 = rad2 * half_thetatrsq
                c1 = twopi_wli*hang2
                c2 = -c1*cs*hang2
                chi = c1 * ( 0.5*(dsum + numpy.cos(2.0 * (numpy.arctan2(float(m), float(l)) - angast))*ddif) ) + c2
                ctfv = -wgh1*numpy.sin(chi)-wgh2*numpy.cos(chi)
            else:
                ctfv = -wgh2
            ctfv2 = ctfv*ctfv
            idx = (l+1)+pow.shape[0]/2*m-1
            cnt += 1
            val = pow.ravel()[idx]
            if hw != 0.0: val *= numpy.exp(hw*res2)
            sum  += val*ctfv2
            sum1 += ctfv2*ctfv2
            sum2 += val*val
    
    if cnt > 0:
        sum /= numpy.sqrt(sum1*sum2)
        #if dast > 0.0: sum -= ddif*ddif/2.0/dast*dast/cnt
    return sum
"""

def ctf_model_full(dz1, sfreq, n, ampcont, cs, voltage, apix):
    ''' Precalculate arrays and values for the CTF model
    
    :Parameters:
    
    sfreq : array
            Pixel offset to be converted to frequency range
    n : int
        Total number of pixels in power spectra
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
             
    :Returns:

    ctf : array
          CTF 1D
    '''
    
    sfreq = sfreq.astype(numpy.float)
    cs *= 1e7
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    w = numpy.arcsin(ampcont)
    factor = numpy.sqrt(cs*lam)
    
    km1 = numpy.sqrt(numpy.sqrt(cs*lam**3)) / (2.0*apix)
    dk = km1/n
    sfreq = sfreq*dk

    sfreq4 = 0.5*numpy.pi*sfreq**4
    sfreq2 = numpy.pi*sfreq**2
    
    dz1/=factor
    
    return numpy.sin( sfreq4 - dz1*sfreq2*1e4 + w )

def ctf_model_spi(dz1, sfreq, n, ampcont, cs, voltage, apix, **extra):
    ''' Precalculate arrays and values for the CTF model
    
    :Parameters:
    
    sfreq : array
            Pixel offset to be converted to frequency range
    dz1 : float
          Defocus value
    n : int
        Total number of pixels in power spectra
    ampcont : float
              Amplitude contrast ratio
    cs : float
        Spherical Abberation (mm)
    voltage : float
              Voltage
    apix : float
           Pixel size
             
    :Returns:

    ctf : array
          CTF 1D
    '''
    
    sfreq = sfreq.astype(numpy.float)
    cs *= 1e7
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    f1    = 1.0 / numpy.sqrt(cs*lam)
    f2    = numpy.sqrt(numpy.sqrt(cs*lam**3))
    max_spat_freq = 1.0 / (2.0 * apix)
    km1   = f2 * max_spat_freq
    dk    = km1 / float(n)
    dz1 = dz1*f1
    
    sfreq *= dk
    qqt = 2.0*numpy.pi*(0.25*sfreq**4 - 0.5*dz1*sfreq**2)
    return (1.0-ampcont)*numpy.sin(qqt)-ampcont*numpy.cos(qqt)

def ctf_model(sfreq, defocus, n, **extra):
    ''' Build a CTF model curve from microscope parameters and given frequency range in pixels
    
    :Parameters:
    
    sfreq : array
            Pixel offset to be converted to frequency range
    defocus : float
              Defocus (A)
    n : int
        Total number of pixels in power spectra
    extra : dict
            Unused keyword arguments
             
    :Returns:
    
    out : array
          CTF model
    '''
    
    defocus/=1e4
    sfreq4, sfreq2, w, factor = ctf_model_precalc(sfreq, n, **extra)
    return ctf_model_calc(defocus/factor, sfreq4, sfreq2, w)

def initial_guess(roo, freq, n, ampcont, cs, voltage, apix, **extra):
    '''
    '''
    
    cs *= 1e7
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    
    km1 = numpy.sqrt(numpy.sqrt(cs*lam**3)) / (2.0*apix)
    dk = km1/n

    sfreq = freq*dk
    sfreq4 = 0.5*numpy.pi*sfreq**4
    sfreq2 = numpy.pi*sfreq**2

    magft = scipy.fftpack.rfft(roo)**2
    p = magft[1:].argmax()+1
    
    return numpy.square(1.0/(sfreq2[p]-sfreq4[p]))
