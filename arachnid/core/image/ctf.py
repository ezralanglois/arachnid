''' CTF utilities including estimate and modeling


.. Created on Dec 4, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy, scipy
import ndimage_utility

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
        val = numpy.max(roo1[b:e])
        roo1[b:e] /=val

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

def local_average(pow, total=32):
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
        segavg[i, :] = pow[b:e].mean(0) #-bg
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

def estimate_defocus_spi(roo, sfreq, n, defocus_start=0.8, defocus_end=7.0, **extra):
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

def estimate_defocus_orig(roo, sfreq, n, defocus_start=1.2, defocus_end=7.0, **extra):
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

def ctf_model_orig(dz1, sfreq, n, ampcont, cs, voltage, apix):
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
