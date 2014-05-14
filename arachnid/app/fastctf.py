''' Automated contrast transfer function estimation

This script (`ara-fastctf`) automatically determines the contrast transfer 
function (CTF) from the 2D power spectra of a micrograph. The 2D power spectra
is estimated using the periodogram method. The CTF is estimated by first
transforming the 2D power spectra to polar space. Then a 1D CTF is estimated
for each angle. This provides an initial estimate of the major and minor defocus
values as well as the angle between the minor and the x-axis. Then, starting with
these initial values, non-linear least squares is used to determine more precise
values using the 2D CTF model.



.. Created on Apr 8, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.image import ndimage_file
from ..core.image import ndimage_utility
from ..core.image import ndimage_interpolate
from ..core.image.ctf import model as ctf_model
from ..core.metadata import spider_params
from ..core.metadata import spider_utility
from ..core.metadata import format_utility
from ..core.metadata import format
from ..core.metadata import selection_utility
from ..core.parallel import mpi_utility
from ..core.util import plotting
import scipy.optimize
import scipy.signal
import warnings
import logging
import numpy
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, use_8bit=False, **extra):#, neig=1, nstd=1.5
    '''Concatenate files and write to a single output file
        
    :Parameters:
        
        filename : str 
                   Filename for input image
        id_len : int
                 Maximum length of the SPIDER ID
        use_8bit : bool
                   Write out space-saving 8-bit 2D diagnostic power spectra image
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    id = spider_utility.spider_id(filename, id_len)
    spider_utility.update_spider_files(extra, id, 'pow_file')
    pow_file=extra['pow_file']
    
    _logger.debug("Generate power spectra")
    pow = generate_powerspectra(filename, **extra)
    #pow += pow.min()+1
    #pow = numpy.log(pow)
    
    defu, defv, defa, error, beg, end, window = estimate_defocus_2D(pow, input_filename=filename, **extra)
    vals=[id, defu, defv, defa, (defu+defv)/2.0, numpy.abs(defu-defv), error]
    
    _logger.debug("Defocus=%f, %f, %f, %f"%(defu, defv, defa, error))
    
    if pow_file != "":
        pow = power_spectra_model_range(pow, defu, defv, defa, beg, end, window, **extra)
        #pow = power_spectra_model(pow, defu, defv, defa, **extra)
        if use_8bit:
            #os.unlink(spi.replace_ext(output_pow))
            ndimage_file.write_image_8bit(pow_file, pow, equalize=True, header=dict(apix=extra['apix']))
        else: ndimage_file.write_image(pow_file, pow, header=dict(apix=extra['apix']))
    
    # Todo:
    # B-factor
    # Selection
    
    
    return filename, numpy.asarray(vals)

def power_spectra_model_range(pow, defu, defv, defa, beg, end, bswindow, ampcont, cs, voltage, apix, bfactor=0, out=None, tdv=0.0, bs=False, mask_pow=False, **extra):
    ''' Generate model for a specific range of rings
    
    :Parameters:
        
        pow : array
              Image of 2D power spectra
        defu : float
               Defocus on minor axis in angstroms
        defv : float
               Defocus on major axis in angstroms
        defa : float
               Astigmatism angle in degrees between x-axis and minor defocus axis
        beg : int
              Starting ring
        end : int
              Last ring
        bswindow : int
                 Size of window for background subtraction
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
        out : array
              Image of 2D power spectra with model on left and data on right
        extra : dict
                Unused keyword arguments
        
    :Returns:
        
        out : array
              Image of 2D power spectra with model on left and data on right
              
    '''
    
    mask = ndimage_utility.model_ring(beg, end, pow.shape) < 0.5
    out=pow.copy()
    model = ctf_model.transfer_function_2D_full(pow.shape, defu, defv, defa, ampcont, cs, voltage, apix, bfactor)**2
    if bs:
        pow = subtract_background(pow, bswindow)
    if tdv > 0:
        from skimage.filter import denoise_tv_chambolle as tv_denoise
        pow = tv_denoise(pow, weight=tdv, eps=2.e-4, n_iter_max=200)
    
    out[:, :pow.shape[0]/2] = model[:, :pow.shape[0]/2]
    
    if mask_pow:
        tmask = mask.copy()
        tmask[:, pow.shape[0]/2:]=0
        gmask = numpy.logical_not(mask.copy())
        gmask[:, pow.shape[0]/2:]=0
        out[tmask] = numpy.mean(model[gmask])
    
    out[:, pow.shape[0]/2:] = pow[:, pow.shape[0]/2:]
    
    if mask_pow:
        tmask = mask.copy()
        tmask[:, :pow.shape[0]/2]=0
        gmask = numpy.logical_not(mask.copy())
        gmask[:, :pow.shape[0]/2]=0
        out[tmask] = numpy.mean(model[gmask])
    
    
    out[:, :pow.shape[0]/2]=ndimage_utility.histeq(out[:, :pow.shape[0]/2])
    out[:, pow.shape[0]/2:]=ndimage_utility.histeq(out[:, pow.shape[0]/2:])
    return out

def estimate_defocus_2D(pow, astig_limit=5000.0, **extra):
    '''Estimate the defocus of an image from the 2D power spectra
    
    :Parameters:
        
        pow : array
              Image of 2D power spectra
        astig_limit : float
                      Maximum allowed astigmastism
        extra : dict
                Unused keyword arguments
    
    :Returns:
    
        defu : float
               Defocus on minor axis in angstroms
        defv : float
               Defocus on major axis in angstroms
        defa : float
               Astigmatism angle in degrees between x-axis and minor defocus axis
        error : float
                Error between data and model
        beg : int
              Starting ring
        end : int
              Last ring
        window : int
                 Size of window for background subtraction
    '''
    
    defu, defv, defa = esimate_defocus_range(pow, **extra)
    _logger.debug("Guess(Attempt #1)=%f, %f, %f"%(defu, defv, defa))
    if defu < 0 or numpy.abs(defu-defv) > astig_limit:
        pow2 = ndimage_interpolate.downsample(pow, 2)
        defu, defv, defa = esimate_defocus_range(pow2, **extra)
        _logger.debug("Guess(Attempt #2)=%f, %f, %f"%(defu, defv, defa))
        if defu < 0 or numpy.abs(defu-defv) > astig_limit:
            pow2 = ndimage_interpolate.downsample(pow, 2)
            defu, defv, defa = esimate_defocus_range(pow2, **extra)
            _logger.debug("Guess(Attempt #3)=%f, %f, %f"%(defu, defv, defa))
        orig = extra['bfactor']
        bfactor = 0.01
        while (defu < 0 or numpy.abs(defu-defv) > astig_limit) and bfactor < 1100:
            extra['bfactor']=bfactor
            defu, defv, defa = esimate_defocus_range(pow2, **extra)
            _logger.debug("Guess(Attempt #4 - %f)=%f, %f, %f"%(bfactor, defu, defv, defa))
            bfactor *= 10
        extra['bfactor']=orig
        if defu < 0 or numpy.abs(defu-defv) > astig_limit:
            defu, defv, defa = esimate_defocus_1D(pow2, **extra)
            _logger.debug("Guess(Attempt #5)=%f, %f, %f"%(defu, defv, defa))
    
    '''
    if defu < 0 or numpy.abs(defu-defv) > 5000:
        pow2 = ndimage_interpolate.downsample(pow, 2)
        param = dict(extra)
        param['bin_factor']*=2
        param.update(spider_params.update_params(**param))
        pow2=generate_powerspectra(extra['input_filename'], **param)
        defu, defv, defa = esimate_defocus_range(pow2, **param)
        _logger.debug("Guess(Attempt #4)=%f, %f, %f"%(defu, defv, defa))
    ''' 
    beg, end, window = resolution_range(pow)
    _logger.debug("Mask: %d - %d"%(beg,end))
    #pow1=pow.copy()
    pow = subtract_background(pow, window)
    
    mask = ndimage_utility.model_ring(beg, end, pow.shape)
    mask[:, :mask.shape[0]/2+beg]=0
    mask = numpy.nonzero(mask)
    
    args = (pow, mask, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix'], extra.get('bfactor', 0))
    defu, defv, defa = scipy.optimize.leastsq(model_fit_error_2d,[defu, defv, defa],args=args)[0]
    error = numpy.sqrt(numpy.sum(numpy.square(model_fit_error_2d([defu, defv, defa], *args))))
    return defu, defv, defa, error, beg, end, window
    
def model_fit_error_2d(p, pow, mask, ampcont, cs, voltage, apix, bfactor):
    ''' Estimate the error between the data, 2D power spectra, and model
    
    :Parameters:
        
        p : array
            A 3-element array with defu, defv and defa, i.e.
            defocus on minor and major axis in angstroms along with angle
            between the x-axis and the minor axis in degrees.
        pow : array
              Image of 2D power spectra
        mask : array
               Valid range to compare
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
        
        error : array
                Error for each pixel between data and model
    '''
    
    model = ctf_model.transfer_function_2D_full(pow.shape, p[0], p[1], p[2], ampcont, cs, voltage, apix, bfactor)**2
    #_logger.debug("Defocus=%f - Error=%f"%(p, numpy.sum(numpy.square(model[mask]-pow[mask]))))
    return model[mask].ravel()-pow[mask].ravel()

def resolution_range(pow):
    ''' Determine resolution range for 2d power spectra using heuristics
    
    The following heuristics are used to determine the range:
        
        #. The first minima of the background subtracted 1D
           power spectra is used as the start
        #. Then point where the energy of the 1D background
           subtract power spectra dips below 95% is used as 
           the finish
    
    :Parameters:
    
        pow : array
              Image of 2D power spectra
    
    :Returns:
    
        beg : int
              Starting resolution as a pixel radius
        end : int
              Ending resolution as a pixel radius
        window : int
                 Window size used for background subtraction
    '''
    
    ppow = ndimage_utility.polar_half(pow, rng=(0, pow.shape[1]/2)).copy()
    window = int(ppow.shape[1]*0.08)
    if (window%2)==0: window+=1
    roo = subtract_background(ppow.mean(axis=0), window)
    beg = first_zero(roo)
    end = energy_cutoff(roo[beg:])+beg
    return beg, end, window

def esimate_defocus_1D(pow, **extra):
    '''Estimate the mean defocus
    
    :Parameters:
    
        pow : array
              Image of 2D power spectra
        extra : dict
                Unused keyword arguments
    
    :Returns:
    
        defu : float
               Defocus on minor axis in angstroms
        defv : float
               Defocus on major axis in angstroms
        defa : float
               Astigmatism angle in degrees between x-axis and minor defocus axis
    '''
    
    ppow = ndimage_utility.polar_half(pow, rng=(0, pow.shape[1]/2)).copy()
    raw = ppow.mean(axis=0)
    window = int(raw.shape[0]*0.08)
    if (window%2)==0: window+=1
    roo = subtract_background(raw, window)
    beg = first_zero(roo)
    end = energy_cutoff(roo[beg:])+beg
    defu = estimate_1D(roo, beg, end, **extra)
    return defu, defu, 0.0

def esimate_defocus_range(pow, awindow_size=64, overlap=0.9, **extra):
    '''Estimate the maximum and minimum defocus as well as the angle of astigmatism
    
    This function calculates the polar form of the power spectra. It, then, estimates
    the CTF over locally averaged radial lines representing 1D power spectra. It saves
    the minimum and maximum defocus and the angle between the minimum and the x-axis.
    
    :Parameters:
    
        pow : array
              Image of 2D power spectra
        awindow_size : int
                       Number of neighboring polar 1D power spectra to average
        overlap : float
                  Overlap between successive lines
        extra : dict
                Unused keyword arguments
    
    :Returns:
    
        defu : float
               Defocus on minor axis in angstroms
        defv : float
               Defocus on major axis in angstroms
        defa : float
               Astigmatism angle in degrees between x-axis and minor defocus axis
    '''
    
    ppow = ndimage_utility.polar_half(pow, rng=(0, pow.shape[1]/2)).copy()
    step = max(1, awindow_size*(1.0-overlap))
    rpow = ndimage_utility.rolling_window(ppow, (awindow_size, 0), (step,1))
    raw = rpow.mean(axis=-1)
    defocus=numpy.zeros(len(raw))
    window = int(raw.shape[1]*0.08)
    if (window%2)==0: window+=1
    roo = subtract_background(ppow.mean(axis=0), window)
    beg = first_zero(roo)
    end = energy_cutoff(roo[beg:])+beg
    for i in xrange(len(raw)):
        defocus[i] = estimate_1D(subtract_background(raw[i], window), beg, end, **extra)
    
    min_defocus = defocus.min()
    max_defocus = defocus.max()
    min_index = defocus.argmin()
    if min_defocus < 0:
        idx = numpy.argsort(defocus)
        sel = defocus[idx] > 0
        if numpy.sum(sel) > 0:
            idx = idx[sel]
            min_defocus = defocus[idx[0]]
            min_index = idx[0]
        
    if 1 == 0:
        idx = numpy.argsort(defocus)
        dd = numpy.diff(defocus[idx])
        std = dd.std()
        avg = dd.mean()
        
        if dd[min(idx[0], len(dd)-1)] < (avg-std*2.5):
            min_defocus = defocus[idx[1]]
            min_index = idx[1]
        if dd[min(idx[-1], len(dd)-1)] > (avg+std*2.5):
            max_defocus = defocus[idx[-2]]
    
    ang = 360.0/raw.shape[0]
    return min_defocus, max_defocus, (raw.shape[0]/2-min_index)*ang
    
def first_zero(roo, offset=10):
    ''' Determine the first zero of the 1D, background 
    subtracted power spectra
    
    :Parameters:
    
        roo : array
              1D, background-subtracted power spectra
    
    :Returns:
        
        zero : int
                Pixel index of first zero
    '''
    
    if 1 == 1:
        from ..core.image import peakdetect_1d
        peak = peakdetect_1d.peakdetect(roo, lookahead=5)[1]
        return peak[1][0]
        
    
    roo = roo[offset:]
    zero = numpy.mean(roo[len(roo)-len(roo)/5:])
    minima = []
    idx = numpy.argwhere(roo < zero).squeeze()
    cuts = numpy.argwhere(numpy.diff(idx) > 1).squeeze()
    b = 0
    for c in cuts:
        minima.append(numpy.argmin(roo[b:idx[c+1]])+b)
        b = idx[c+1]
    #n = len(minima)-1 if len(minima) < 2 else 1
    if len(minima)==0:return offset
    if len(minima) > 1 and minima[0]==0: return minima[1]+offset
    return minima[0]+offset

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
    
    roo = numpy.abs(roo)
    return numpy.searchsorted(numpy.cumsum(roo)/roo.sum(), energy)

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
    if roo.ndim==2:
        weightings = numpy.ones((window, window))
        weightings /= weightings.sum()
        bg[:]=scipy.signal.convolve2d(roo, weightings, mode='same', boundary='fill', fillvalue=0)
        return roo-bg
    
    weightings = numpy.ones(window)
    weightings /= weightings.sum()
    bg[off:len(roo)-off]=numpy.convolve(roo, weightings)[window-1:-(window-1)]
    return roo-bg

def denoise():
    '''
    '''
    from skimage.filter import denoise_tv_chambolle as tv_denoise
    
    """
    raw = ndimage_utility.mean_azimuthal(pow)[4:pow.shape[0]/2]
        area = scipy.integrate.trapz(raw, dx=5)
        
        if extra['model_file'] != "":
        _logger.info("Running diagnostic2")
        for i, reg in enumerate([10, 1, 0.1, 0.001, 0.0001, 0.00001]):
            dnpow = tv_denoise(pow, weight=reg, eps=2.e-4, n_iter_max=200)
            _logger.info("Running diagnostic3-%d"%i)
            if i == 0:
                _logger.info("Running diagnostic-plot")
                roo1 = ndimage_utility.mean_azimuthal(dnpow)
                roo2 = dnpow[0, :]
                roo1 = roo1[10:len(roo1)-10]
                roo2 = roo2[10:len(roo1)-10]
                roo1 /= roo1.max()
                roo2 /= roo2.max()
                plotting.plot_lines(extra['model_file'], [roo1, roo2], ['Average', 'Single1'])
                plotting.plot_lines(extra['model_file'], [roo2], ['Single1'], prefix='single_')
                plotting.plot_lines(extra['model_file'], [roo1], ['Average'], prefix='avg_')
            ndimage_file.write_image(extra['model_file'], dnpow, 1+i)
    """

def model_fit_error_1d(p, roo, beg, end, ampcont, cs, voltage, apix, bfactor):
    ''' Compute the error between the 1D power spectra and the model CTF
    
    :Parameters:
        
        p : array
            A a-element array with the current defocus value
        roo : array
              1D, background-subtracted power spectra
        beg : int
              Starting index
        end : int
              Last index
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
    
    :Returns:
    
        error : array
                Error for each value in valid range
    '''
    
    model = ctf_model.transfer_function_1D(len(roo), p, ampcont, cs, voltage, apix, bfactor)**2
    # problem likely due to rotational average code - mean_azimuthal (polar alleviates)
    #model = ctf_model.transfer_function_2D(len(roo)*2, p, ampcont, cs, voltage, apix, bfactor)**2
    #model = ndimage_utility.polar_half(model, rng=(0, model.shape[1]/2)).copy()
    #model = model.mean(axis=0)
    #model = ndimage_utility.mean_azimuthal(model)[0:model.shape[0]/2]
    #_logger.debug("Defocus=%f - Error=%f"%(p, numpy.sum(numpy.square(model[beg:end]-roo[beg:end]))))
    return model[beg:end]-roo[beg:end]

def estimate_1D(roo, beg, end, ampcont, cs, voltage, apix, bfactor=0.0, defocus_start=0.1, defocus_end=8.0, **extra):
    ''' Find the defocus of the image from the background-subtracted 1D power spectra
    
    :Parameters:
        
        roo : array
              1D, background-subtracted power spectra
        beg : int
              Starting ring
        end : int
              Last ring
        ampcont : float
                  Amplitude contrast in percent
        cs : float
             Spherical abberation in mm
        voltage : float
                  Electron energy in kV
        apix : float
               Pixel size
        bfactor : float
                  Fall off in angstroms^2
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        defocus : float
                  Defocus of image
    '''
    
    best=(1e20, None)
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        err = numpy.sum(numpy.square(model_fit_error_1d(p*1e4, roo, beg, end, ampcont, cs, voltage, apix, bfactor)))
        if err < best[0]: best = (err, p*1e4)
    p0=[best[1]]
    dz1, = scipy.optimize.leastsq(model_fit_error_1d,p0,args=(roo, beg, end, ampcont, cs, voltage, apix, bfactor))[0]
    return dz1
    
def generate_powerspectra(filename, bin_factor, window_size, overlap, pad=1, offset=0, from_power=False, **extra):
    ''' Generate a power spectra using a perdiogram
    
    :Parameters:
        
        filename : str
                   Input filename
        bin_factor : float
                    Decimation factor
        window_size : int
                      Perdiogram window size
        overlap : float
                  Amount of overlap between windows
        pad : int
              Number of times to pad the perdiogram
        offset : int
                 Offset from the edge of the micrograph
        from_power : bool
                     Is the input file already a power spectra
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        pow : array
              2D power spectra
    '''
    
    if from_power: return ndimage_file.read_image(filename)
    mic = ndimage_file.read_image(filename)
    #if bin_factor > 1.0: mic = ndimage_interpolate.resample_fft(mic, bin_factor, pad=3)
    if bin_factor > 1.0: mic = ndimage_interpolate.downsample(mic, bin_factor)
    pow = ndimage_utility.perdiogram(mic, window_size, pad, overlap, offset)
    return pow

def _perdiogram(mic, window_size=256, pad=1, overlap=0.5, offset=0.1, shift=True, feature_size=8):
    '''
    '''
    
    if offset > 0 and offset < 1.0: offset = int(offset*mic.shape[0])
    step = max(1, window_size*overlap)
    rwin = ndimage_utility.rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
    rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
    gwin=[]
    for win in rwin:
        win1 = win.copy()
        win1 = ndimage_utility.normalize_standard(win1)
        fstep = max(1, feature_size*overlap)
        fwin = ndimage_utility.rolling_window(win1, (feature_size, feature_size), (fstep, fstep))
        if numpy.std(fwin, axis=0).min() > 0.5: gwin.append(win)
    _logger.debug("Using %d of %d windows"%(len(gwin), len(rwin)))
    return ndimage_utility.powerspec_avg(gwin, pad, shift)

def plot_scatter(output, x, x_label, y, y_label, dpi=72):
    ''' Plot a histogram of the distribution
    '''
    
    pylab=plotting.pylab
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    
    '''
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
    plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    '''
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_"+y_label.lower().replace(' ', '_')+"_", ext="png"), dpi=dpi)
    
def plot_histogram(output, vals, x_label, th=None, dpi=72):
    ''' Plot a histogram of the distribution
    '''
    
    pylab=plotting.pylab
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    vals = ax.hist(vals, bins=numpy.sqrt(len(vals)))
    if th is not None:
        h = pylab.gca().get_ylim()[1]
        pylab.plot((th, th), (0, h))
    pylab.xlabel(x_label)
    pylab.ylabel('Number of Micrographs')
    
    '''
    index = select[plotting.nonoverlapping_subset(ax, eigs[select, 0], eigs[select, 1], radius, 100)].ravel()
    iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
    plotting.plot_images(fig, iter_single_images, eigs[index, 0], eigs[index, 1], image_size, radius)
    '''
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_", ext="png"), dpi=dpi)

def initialize(files, param):
    # Initialize global parameters for the script
    
    warnings.simplefilter('error', UserWarning)
    spider_params.read(param['param_file'], param)
    if mpi_utility.is_root(**param):
        _logger.info("Input: %s"%( "Micrograph" if not param['from_power'] else "Power spec"  ))
        _logger.info("Bin-factor: %f"%param['bin_factor'])
        _logger.info("Window size: %f"%param['window_size'])
        if 'pad' in param: _logger.info("Pad: %f"%param['pad'])
        _logger.info("Overlap: %f"%param['overlap'])
        #_logger.info("Plotting disabled: %d"%plotting.is_plotting_disabled())
        _logger.info("Microscope parameters")
        _logger.info(" - Voltage: %f"%param['voltage'])
        _logger.info(" - CS: %f"%param['cs'])
        _logger.info(" - AmpCont: %f"%param['ampcont'])
        _logger.info(" - Pixel size: %f"%param['apix'])
        if param['pow_file'] != "":
            try:
                os.makedirs(os.path.dirname(param['pow_file']))
            except: pass
            _logger.info("Writing power spectra to %s"%param['pow_file'])
        try:
            defvals = format.read(param['output'], map_ids=True)
        except:
            param['output_offset']=0
        else:
            saved=[]
            for filename in param['finished']:
                id = spider_utility.spider_id(filename)
                if id not in defvals: 
                    files.append(filename)
                    continue
                saved.append(defvals[id])
            if len(saved) > 0: format.write(param['output'], saved)
            param['output_offset']=len(saved)
        
        if param['cs'] == 0.0:
            _logger.info("Using CTF model appropriate for 0 CS")
        if param['selection_file'] != "":
            select = format.read(param['selection_file'], numeric=True)
            files = selection_utility.select_file_subset(files, select, param.get('id_len', 0), len(param['finished']) > 0)
        param['defocus_header']="id,defocus_u,defocus_v,astig_ang,defocus_avg,astig_mag,error".split(",")
        param['defocus_arr'] = numpy.zeros((len(files), 7))
    return files

def reduce_all(filename, file_completed, defocus_arr, output_offset, output, defocus_header, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, defocus_vals = filename
    if len(defocus_vals) > 0:
        defocus_arr[file_completed-1, :len(defocus_vals)]=defocus_vals
        mode = 'a' if (file_completed+output_offset) > 1 else 'w'
        format.write(output, defocus_vals.reshape((1, defocus_vals.shape[0])), format=format.spiderdoc, 
                             header=defocus_header, mode=mode, write_offset=file_completed+output_offset)
    return filename

def finalize(files, defocus_arr, output, dpi=300, **extra):
    ''' Write out plots summarizing CTF features of the input images
    
    These plots include
        
        - Mean defocus histogram
        - Astigmatism magnitude histogram
        - Mean defocus vs. Error scatter plot
        - Mean defocus vs. Astigmatism angle scatter plot (for magnitude > 2000 angstroms)
        
    :Parameters:
        
        files : list
                List of input filenames
        defocus_arr : array
                      Array of CTF parameters for each input file, with columns:
                      (id,defocus_u,defocus_v,astig_ang,defocus_avg,astig_mag,error)
        output : str
                 Output filename 
        dpi : int
              Resolution of plot in dots per inch
        extra : dict
                Unused keyword arguments
    '''
    
    if len(files) > 0:
        defocus = (defocus_arr[:, 1]+defocus_arr[:, 2])/2.0
        magnitude = numpy.abs((defocus_arr[:, 1]-defocus_arr[:, 2])/2.0)
        plot_histogram(output, defocus, 'Defocus', dpi=dpi)
        plot_histogram(output, magnitude, 'Astigmatism', dpi=dpi)
        plot_scatter(output, defocus, 'Defocus', defocus_arr[:, 4], 'Error', dpi=dpi)
        sel = magnitude > 2000
        if numpy.sum(sel) > 0:
            plot_scatter(output, defocus[sel], 'Defocus', defocus_arr[sel, 3], 'Angle', dpi=dpi)
        
    
    # Plots
    # 1. Defocus histogram
    # 2. Error scatter
    # 3. Falloff scatter
    
    _logger.info("Completed")
"""
def supports(files, **extra):
    ''' Test if this module is required in the project workflow
    
    :Parameters:
        
        files : list
                List of filenames to test
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        flag : bool
               True if this module should be added to the workflow
    '''
    
    return True
"""

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "CTF", "Options to control contrast transfer function estimation",  id=__name__)
    group.add_option("", window_size=256, help="Size of the window for the power spec (pixels)")
    group.add_option("", overlap=0.5, help="Amount of overlap between windows")
    group.add_option("", offset=0, help="Offset from the edge of the micrograph (pixels)")
    group.add_option("", from_power=False, help="Input is a powerspectra not a micrograph")
    group.add_option("", awindow_size=8, help="Window size for polar average")
    group.add_option("", tdv=0.0,        help="Regularizstion for total variance denosing for output diagnostic power spectra")
    group.add_option("", bs=False,        help="Background subtract for output diagnostic power spectra")
    group.add_option("", mask_pow=False,  help="Mask diagnostic power spectra")
    group.add_option("", bfactor=0.0,     help="B-factor for fitting")
    
    
    
    #group.add_option("", pad=2.0, help="Number of times to pad the power spec") # - Caused all sorts of issues in 2D
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--micrograph-files", input_files=[],     help="List of filenames for the input micrographs, e.g. mic_*.mrc", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", "--ctf-file",         output="",          help="Output filename for ctf file", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("",   pow_file="",                              help="Output filename for power spectra file - Display model in first half", gui=dict(filetype="save"), required_file=False)
        pgroup.add_option("-s", selection_file="",                        help="Selection file for a subset of good micrographs", gui=dict(filetype="open"), required_file=False)
        spider_params.setup_options(parser, pgroup, True)
        parser.change_default(log_level=3, bin_factor=2)

def flags():
    ''' Get flags the define the supported features
    
    Returns:
    
        flags : dict
                Supported features
    '''
    
    return dict(description = '''Automated contrast transfer function estimation (fastctf)
                        
                        Example:
                         
                        $ %prog mic.dat -o ctf.dat -p params.dat
                      ''',
                supports_MPI=True, 
                supports_OMP=True,
                use_version=True)

def main():
    '''Main entry point for this script
    
    .. seealso:: 
    
        arachnid.core.app.program.run_hybrid_program
    
    '''
    program.run_hybrid_program(__name__)

if __name__ == "__main__": main()

