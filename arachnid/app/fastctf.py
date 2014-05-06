'''
.. Created on Apr 8, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.app import program
from ..core.image import ndimage_file
from ..core.image import ndimage_utility
from ..core.image import ndimage_interpolate
from ..core.image.ctf import model as ctf_model, estimate2d, estimate1d
from ..core.metadata import spider_params
from ..core.metadata import spider_utility
from ..core.metadata import format_utility
from ..core.metadata import format
from ..core.metadata import selection_utility
from ..core.parallel import mpi_utility
from ..core.util import plotting
import warnings
import logging
import numpy
#import scipy.special
#import scipy.misc
#import scipy.integrate
import scipy.optimize
import scipy.signal
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, id_len=0, **extra):#, neig=1, nstd=1.5
    '''Concatenate files and write to a single output file
        
    :Parameters:
        
        filename : str 
                   Filename for input image
        output : str
                 Filename for output file
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    id = spider_utility.spider_id(filename, id_len)
    spider_utility.update_spider_files(extra, id, 'pow_file', 'diagnostic_file')
    diagnostic_file=extra['diagnostic_file']
    
    _logger.debug("Generate power spectra")
    pow = generate_powerspectra(filename, **extra)
    
    defu, defv, defa, error = estimate_defocus_2D(pow, **extra)
    vals=[id, defu, defv, defa, (defu+defv)/2.0, numpy.abs(defu-defv), error]
    
    _logger.debug("Defocus=%f, %f, %f, %f"%(defu, defv, defa, error))
    
    if diagnostic_file != "":
        pow = power_spectra_model(pow, defu, defv, defa, **extra)
        ndimage_file.write_image(diagnostic_file, pow)
    
    
    
    #model = ctf_model_array(defocus, len(roo), **extra)
    #plotting.plot_lines(extra['diagnostic_file'], [roo[beg:end], model[beg:end]], ['ROO', 'Model'])
    
    # B-factor
    # Selection
    
    
    return filename, numpy.asarray(vals)

def power_spectra_model(pow, defu, defv, defa, ampcont, cs, voltage, apix, bfactor=0, **extra):
    '''
    '''
    
    pow=pow.copy()
    model = ctf_model.transfer_function_2D_full(pow.shape, defu, defv, defa, ampcont, cs, voltage, apix, bfactor)**2
    pow[:, :pow.shape[0]/2]=ndimage_utility.histeq(model[:, :pow.shape[0]/2])
    pow[:, pow.shape[0]/2:]=ndimage_utility.histeq(pow[:, pow.shape[0]/2:])
    return pow

def estimate_defocus_2D(pow, **extra):
    '''
    '''
    
    defu, defv, defa = esimate_defocus_range(pow, **extra)
    _logger.debug("Guess=%f, %f, %f"%(defu, defv, defa))
    if defu < 0:
        pow2 = ndimage_interpolate.downsample(pow, 2)
        defu, defv, defa = esimate_defocus_range(pow2, **extra)
        _logger.debug("Guess=%f, %f, %f"%(defu, defv, defa))
        
    beg, end, window = resolution_range(pow)
    _logger.debug("Mask: %d - %d"%(beg,end))
    pow1=pow.copy()
    pow = subtract_background(pow, window)
    
    mask = ndimage_utility.model_ring(beg, end, pow.shape)
    mask[:, :mask.shape[0]/2+beg]=0
    mask = numpy.nonzero(mask)
    
    if 1 == 0:
        rmin = estimate1d.resolution(beg, pow.shape[0]/2, extra['apix'])
        rmax = estimate1d.resolution(end, pow.shape[0]/2, extra['apix'])
        vals = estimate2d.fit_ctf(pow1, defu, defv, defa, rmin, rmax, **extra)
        _logger.info("CTFFIND3=%f, %f, %f"%tuple(vals))
    
    args = (pow, mask, extra['ampcont'], extra['cs'], extra['voltage'], extra['apix'], extra.get('bfactor', 0))
    defu, defv, defa = scipy.optimize.leastsq(model_fit_error_2d,[defu, defv, defa],args=args)[0]
    error = numpy.sqrt(numpy.sum(numpy.square(model_fit_error_2d([defu, defv, defa], *args))))
    return defu, defv, defa, error
    
def model_fit_error_2d(p, pow, mask, ampcont, cs, voltage, apix, bfactor):
    '''
    '''
    
    model = ctf_model.transfer_function_2D_full(pow.shape, p[0], p[1], p[2], ampcont, cs, voltage, apix, bfactor)**2
    #_logger.debug("Defocus=%f - Error=%f"%(p, numpy.sum(numpy.square(model[mask]-pow[mask]))))
    return model[mask].ravel()-pow[mask].ravel()

def estimate_defocus_2D_no_astig(pow, **extra):
    '''
    '''
    
    beg, end, window = resolution_range(pow)
    pow = subtract_background(pow, window)
    return estimate_2D_no_astig(pow, beg, end, **extra)

def estimate_defocus_1D(pow, **extra):
    '''
    '''
    
    ppow = ndimage_utility.polar_half(pow, rng=(0, pow.shape[1]/2)).copy()
    raw = ppow.mean(axis=0)
    window = int(len(raw)*0.08)
    if (window%2)==0: window+=1
    #_logger.debug("Subtract background with window: %d"%window)
    roo = subtract_background(raw, window)
    beg = first_zero(roo)
    end = energy_cutoff(roo[beg:])+beg
    return estimate_1D(roo, beg, end, **extra)

def resolution_range(pow):
    '''
    '''
    
    ppow = ndimage_utility.polar_half(pow, rng=(0, pow.shape[1]/2)).copy()
    window = int(ppow.shape[1]*0.08)
    if (window%2)==0: window+=1
    roo = subtract_background(ppow.mean(axis=0), window)
    beg = first_zero(roo)
    end = energy_cutoff(roo[beg:])+beg
    return beg, end, window

def esimate_defocus_range(pow, awindow_size=64, overlap=0.9, **extra):
    '''
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
    
    ang = 360.0/raw.shape[0]
    return defocus.min(), defocus.max(), (raw.shape[0]/2-defocus.argmin())*ang
    
def first_zero(roo, **extra):
    ''' Determine the first zero of the CTF
    '''
    
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
        
        if extra['diagnostic_file'] != "":
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
                plotting.plot_lines(extra['diagnostic_file'], [roo1, roo2], ['Average', 'Single1'])
                plotting.plot_lines(extra['diagnostic_file'], [roo2], ['Single1'], prefix='single_')
                plotting.plot_lines(extra['diagnostic_file'], [roo1], ['Average'], prefix='avg_')
            ndimage_file.write_image(extra['diagnostic_file'], dnpow, 1+i)
    """
    
def ctf_model_array(p, n, ampcont, cs, voltage, apix, bfactor=0, **extra):
    '''
    '''
    
    model = ctf_model.transfer_function_1D(n, p, ampcont, cs, voltage, apix, bfactor)
    return model**2

def model_fit_error_1d_bfactor(p, roo, beg, end, ampcont, cs, voltage, apix):
    '''
    '''
    
    model = ctf_model.transfer_function_1D(len(roo), p[0], ampcont, cs, voltage, apix, p[1])**2
    return model[beg:end]-roo[beg:end]

def estimate_1D_bfactor(roo, beg, end, defocus, ampcont, cs, voltage, apix, bfactor=0.0, bfactor_start=0.2, bfactor_end=8.0, **extra):
    '''
    '''
    
    best=(1e20, None)
    
    for p in scipy.logspace(0.1, 5, 50):
        err = numpy.sum(numpy.square(model_fit_error_1d(defocus, roo, beg, end, ampcont, cs, voltage, apix, p)))
        if err < best[0]: best = (err, p)
    #_logger.debug("-------Best: %s"%str(best))
    bfactor=best[1]
    dz1, bfactor= scipy.optimize.leastsq(model_fit_error_1d_bfactor,[defocus,bfactor],args=(roo, beg, end, ampcont, cs, voltage, apix))[0]
    return dz1, bfactor

def model_fit_error_1d(p, roo, beg, end, ampcont, cs, voltage, apix, bfactor):
    '''
    '''
    
    model = ctf_model.transfer_function_1D(len(roo), p, ampcont, cs, voltage, apix, bfactor)**2
    # problem likely due to rotational average code - mean_azimuthal (polar alleviates)
    #model = ctf_model.transfer_function_2D(len(roo)*2, p, ampcont, cs, voltage, apix, bfactor)**2
    #model = ndimage_utility.polar_half(model, rng=(0, model.shape[1]/2)).copy()
    #model = model.mean(axis=0)
    #model = ndimage_utility.mean_azimuthal(model)[0:model.shape[0]/2]
    #_logger.debug("Defocus=%f - Error=%f"%(p, numpy.sum(numpy.square(model[beg:end]-roo[beg:end]))))
    return model[beg:end]-roo[beg:end]

def estimate_1D(roo, beg, end, ampcont, cs, voltage, apix, bfactor=0.0, defocus_start=0.2, defocus_end=8.0, **extra):
    '''
    '''
    
    best=(1e20, None)
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        err = numpy.sum(numpy.square(model_fit_error_1d(p*1e4, roo, beg, end, ampcont, cs, voltage, apix, bfactor)))
        if err < best[0]: best = (err, p*1e4)
    #_logger.debug("-------Best: %s"%str(best))
    p0=[best[1]]
    dz1, = scipy.optimize.leastsq(model_fit_error_1d,p0,args=(roo, beg, end, ampcont, cs, voltage, apix, bfactor))[0]
    return dz1

def ctf_model_array_2d(p, shape, ampcont, cs, voltage, apix, bfactor=0, **extra):
    '''
    '''
    
    model = ctf_model.transfer_function_2D(shape, p, ampcont, cs, voltage, apix, bfactor)
    return model**2

def model_fit_error_2d_no_astig(p, pow, mask, ampcont, cs, voltage, apix, bfactor):
    '''
    '''
    
    model = ctf_model.transfer_function_2D(pow.shape, p, ampcont, cs, voltage, apix, bfactor)**2
    #_logger.debug("Defocus=%f - Error=%f"%(p, numpy.sum(numpy.square(model[mask]-pow[mask]))))
    return model[mask].ravel()-pow[mask].ravel()

def estimate_2D_no_astig(pow, beg, end, ampcont, cs, voltage, apix, bfactor=0.0, defocus_start=0.15, defocus_end=8.0, **extra):
    '''
    '''
    
    mask = ndimage_utility.model_ring(beg, end, pow.shape)
    mask[:, :mask.shape[0]/2+beg]=0
    mask = numpy.nonzero(mask)
    best=(1e20, None)
    for p in numpy.arange(defocus_start, defocus_end, 0.1, dtype=numpy.float):
        err = numpy.sum(numpy.square(model_fit_error_2d_no_astig(p*1e4, pow, mask, ampcont, cs, voltage, apix, bfactor)))
        if err < best[0]: best = (err, p*1e4)
    #_logger.debug("-------Best: %s"%str(best))
    p0=[best[1]]
    dz1, = scipy.optimize.leastsq(model_fit_error_2d_no_astig,p0,args=(pow, mask, ampcont, cs, voltage, apix, bfactor))[0]
    return dz1
    
def generate_powerspectra(filename, bin_factor, window_size, overlap, pad=1, offset=0, from_power=False, pow_file="", **extra):
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
        from_power : bool
                     Is the input file already a power spectra
    
    :Returns:
        
        pow : array
              2D power spectra
    '''
    
    if from_power: return ndimage_file.read_image(filename)
    mic = ndimage_file.read_image(filename)
    #if bin_factor > 1.0: mic = ndimage_interpolate.resample_fft(mic, bin_factor, pad=3)
    if bin_factor > 1.0: mic = ndimage_interpolate.downsample(mic, bin_factor)
    pow = ndimage_utility.perdiogram(mic, window_size, pad, overlap, offset)
    if pow_file != "": ndimage_file.write_image(pow_file, pow)
    return pow

def perdiogram(mic, window_size=256, pad=1, overlap=0.5, offset=0.1, shift=True, feature_size=8):
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
        if param['diagnostic_file'] != "":
            try:
                os.makedirs(os.path.dirname(param['diagnostic_file']))
            except: pass
            _logger.info("Writing diagnostic power spectra images to stack: %s"%param['diagnostic_file'])
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
    '''
    '''
    
    if len(files) > 0:
        defocus = (defocus_arr[:, 1]+defocus_arr[:, 2])/2.0
        plot_histogram(output, defocus, 'Defocus', dpi=dpi)
        plot_histogram(output, numpy.abs((defocus_arr[:, 1]-defocus_arr[:, 2])/2.0), 'Astigmatism', dpi=dpi)
        plot_scatter(output, defocus, 'Defocus', defocus_arr[:, 4], 'Error', dpi=dpi)
    
    # Plots
    # 1. Defocus histogram
    # 2. Error scatter
    # 3. Falloff scatter
    
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", window_size=256, help="Size of the window for the power spec (pixels)")
    #group.add_option("", pad=2.0, help="Number of times to pad the power spec")
    group.add_option("", overlap=0.5, help="Amount of overlap between windows")
    group.add_option("", offset=0, help="Offset from the edge of the micrograph (pixels)")
    group.add_option("", from_power=False, help="Input is a powerspectra not a micrograph")
    group.add_option("", awindow_size=8, help="Window size for polar average")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--micrograph-files", input_files=[],     help="List of filenames for the input micrographs, e.g. mic_*.mrc", required_file=True, gui=dict(filetype="open"), regexp=spider_utility.spider_searchpath)
        pgroup.add_option("-o", "--ctf-file",         output="",          help="Output filename for ctf file", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("",   pow_file="",                              help="Output filename for power spectra file", gui=dict(filetype="save"), required_file=False)
        pgroup.add_option("",   diagnostic_file="",                       help="Output filename for stack of diagnostic images", gui=dict(filetype="save"), required_file=False)

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

