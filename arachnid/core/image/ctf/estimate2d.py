''' Based on CTFFIND3

.. Created on Nov 25, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from arachnid.core.app import tracing
from .. import ndimage_interpolate
import numpy, scipy
import scipy.optimize


import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from ..util import _ctf
    _ctf;
except:
    _ctf=None
    tracing.log_import_error('Failed to load _ctf.so module', _logger)

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
        pow_sm = ndimage_interpolate.downsample(pow, float(pow.shape[0])/pre_decimate)
    else: pow_sm = pow
    thetatr = wl/(apix*min(pow_sm.shape))
    pow_sm = pow_sm.copy()
    pow_sm = pow_sm[:pow_sm.shape[0]/2, :].copy()
    pow_sm = pow_sm.T.copy()
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
        thetatr = wl/(apix*min(pow.shape))
        pow = pow.copy()
        pow = pow[:pow.shape[0]/2, :].copy()
        pow = pow.T.copy()
        smooth_2d(pow, rmin)
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

def ctf_2d(pow, dfmid1, dfmid2, angast, ampcont, cs, voltage, rmin, rmax, apix=None, xmag=None, res=None, **extra):
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
    
    if rmin < rmax: rmin, rmax = rmax, rmin
    if rmin > 50: rmin = 50.0
    rmin = apix/rmin
    rmax = apix/rmax
    rmin2, rmax2 = rmin**2,rmax**2
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
            res2 = (float(l)/pow.shape[0])**2 + (float(m)/pow.shape[1])**2
            if res2 <= rmax2 and res2 >= rmin2:
                pow.ravel()[j+pow.shape[1]*i] = ctf_2d_value(l, m, dfmid1, dfmid2, angast, cs, wl, ampcont, thetatr)**2
            else: pow.ravel()[j+pow.shape[1]*i]=0

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