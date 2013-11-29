'''
.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from .. import ndimage_utility
import numpy
import scipy.special
import scipy.misc
import scipy.linalg
import heapq
DEBUG=1

######################################################################
# Fast Def
######################################################################


def normalize_wb(img, rmin, rmax, sel):
    '''
    '''
    
    assert(numpy.sum(sel)>0)
    img_mod = img.copy()
    rang = (rmax-rmin)/2.0
    freq2 = float(img.shape[0])/(rang/1)
    freq1 = float(img.shape[0])/(rang/15)
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
    quality_vals = -((ormod/ormod[imax,jmax])*(max_levels-1)).round()# pos or neg?
    
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
    heapq.heappush(pqueue, (quality_vals[imax, jmax], imax, jmax)) 
    pred, cor, norm = 0.0, 0.0, 0.0
    while len(pqueue)>0:
        i, j = heapq.heappop(pqueue)[1:]
        ind_i[:] = [i-1, i-1, i-1, i,   i  , i+1, i+1, i+1]
        ind_j[:] = [j-1, j,   j+1, j-1, j+1, j-1, j,   j+1]
        for k in xrange(8):
            ni, nj = ind_i[k], ind_j[k]
            if (ni-size)>0 and (nj-size)>0 \
                and (ni+size)<(processed.shape[0]-1) \
                and (nj+size)<(processed.shape[1]-1) \
                and not processed[ni, nj] and \
            ormod[ni,nj]> min_quality:
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
                heapq.heappush(pqueue, (quality_vals[ni, nj], ni, nj))
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

"""
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
"""

def unwrapped_phase(pow, rmin, rmax, apix, eps_phase=2.0, window_phase=10):
    '''
    '''
    
    if rmin < rmax: rmin, rmax = rmax, rmin
    min_freq = apix/rmin
    max_freq = apix/rmax
    rmin = min_freq*pow.shape[0]
    rmax = max_freq*pow.shape[0]
    
    roi = ndimage_utility.model_ring(rmin, rmax, pow.shape, dtype=numpy.bool)
    x,y = ndimage_utility.grid_image(pow.shape)
    dir = -numpy.arctan2(x, y)
    img_norm, img_mod = normalize_wb(pow, rmin, rmax, roi)
    img_mod /= img_mod.max()
    sph = ndimage_utility.spiral_transform(img_norm)
    wphase, ormod = wrapped_phase(sph, dir, img_norm, roi)
    return unwrapping(wphase, ormod, eps_phase, window_phase)

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
        from .. import ndimage_file
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