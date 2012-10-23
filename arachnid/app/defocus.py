''' Estimate the CTF parameters from a micrograph or a stack of particles


.. Created on Sep 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import tracing
tracing;
from ..core.metadata import format, spider_params, spider_utility, format_utility
format;
from ..core.image import ndimage_utility, ndimage_file, eman2_utility
from ..core.image.ndplot import pylab
import numpy, logging, os, scipy.optimize
from ..pyspider import resolution

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, id_len=0, **extra):
    '''Concatenate files and write to a single output file
    
    :Parameters:
        
    filename : str
               Input filename
    output : str
             Filename for output file
    id_len : int
             Maximum length of the ID
    extra : dict
            Unused key word arguments
      
    :Returns:
        
    filename : string
          Current filename
    CTF : numpy.ndarray
          CTF parameters
    '''
    
    id = spider_utility.spider_id(filename)
    output = spider_utility.spider_filename(output, id)
    
    _logger.info("Processing2 %s"%filename)
    if 1 == 0:
        esimate_signal(filename, output, **extra)
        defocus = 0
    else:
        _logger.info("Estimating power spectra")
        powerspec = powerspectra(filename, **extra)
        #spowerspec = ndimage_file.read_image(spider_utility.spider_filename("secA_prj/cluster/data/local/pow/pow_20483.dat", id))
        #numpy.testing.assert_allclose(spowerspec, powerspec)
        '''
         x: array([[ 0.09776067,  0.0975919 ,  0.09716196, ...,  0.09666254,
         0.09716196,  0.0975919 ],
       [ 0.09737622,  0.09700554,  0.09638637, ...,  0.09702981,...
 y: array([[ 751.73982509,  752.446702  ,  755.05830071, ...,  760.38431899,
         755.05830071,  752.446702  ],
       [ 752.53202192,  761.48459767,  767.59839725, ...,  757.48932337,...

        '''
        write_powerspectra(output, powerspec, **extra)
        _logger.info("Estimating defocus")
        defocus = esimate_defocus(powerspec, **extra)
        _logger.info("Defocus=%f"%defocus)
        plot_ctf(output, powerspec, defocus, **extra)
    
    # Calculate astig
    
    return filename, numpy.asarray((id, defocus, 0, 0, 0))

def esimate_signal(filename, output, apix, window_size=500, overlap=0.5, pad=2, **extra):
    ''' Estimate the signal based on two power spectra
    
    Use ellipse as a contraint
    
    1. modify the ring width
    2. invert mask
    3. rot avg
    4. correlation between model and average - 1d to 2d
    
    noise window search
    '''
    step = max(1, window_size*overlap)
    rwin = ndimage_utility.rolling_window(read_micrograph(filename, **extra), (window_size, window_size), (step, step))
    rwin1 = rwin[::2]
    rwin2 = rwin[1::2]
    if rwin1.shape[0] > rwin2.shape[0]: rwin1 = rwin1[:rwin2.shape[0]]
    elif rwin2.shape[0] > rwin1.shape[0]: rwin2 = rwin2[:rwin1.shape[0]]
    avg1 = ndimage_utility.powerspec_avg(rwin1.reshape((rwin1.shape[0]*rwin1.shape[1], rwin1.shape[2], rwin1.shape[3])), pad)
    avg2 = ndimage_utility.powerspec_avg(rwin2.reshape((rwin2.shape[0]*rwin2.shape[1], rwin2.shape[2], rwin2.shape[3])), pad)
    
    
    #1. astig (remove? or correct?)
    #2. Signal fall off - low resolution
    #     compare average to model
    #     global model for fall off
    #3. Drift
    #4. No rings
    
    ravg1 = ndimage_utility.mean_azimuthal(avg1)
    ravg2 = ndimage_utility.mean_azimuthal(avg2)
    
    freq = numpy.arange(0, ravg1.shape[0], dtype=numpy.float)/(2.0*ravg1.shape[0])
    #float(i)/float(2*inc);
    
    #scipy.spatial.distance.correlation(ravg1, ravg2)
    #ravg1m = ravg1-numpy.mean(ravg1) 
    #ravg2m = ravg2-numpy.mean(ravg2)
    #fsc = 1.0 - (ravg1m*ravg2m) / ( numpy.sqrt(numpy.sum(numpy.square(ravg1m))) * numpy.sqrt(numpy.sum(numpy.square(ravg2m))) )
    fsc = numpy.square(ravg1-ravg2)
    fsc /= numpy.max(fsc)
    resolution.plot_fsc(format_utility.add_prefix(output, 'rotavg_'), freq, fsc, apix)
    
    fsc = numpy.square(ravg1-ravg1[::-1])
    fsc /= numpy.max(fsc)
    resolution.plot_fsc(format_utility.add_prefix(output, 'rev_'), freq, fsc, apix)

#
def esimate_signal_old(filename, output, apix, window_size=500, overlap=0.5, pad=2, **extra):
    ''' Estimate the signal based on two power spectra
    
    Use ellipse as a contraint
    
    1. modify the ring width
    2. invert mask
    3. rot avg
    4. correlation between model and average - 1d to 2d
    
    noise window search
    '''
    
    step = max(1, window_size*overlap)
    rwin = ndimage_utility.rolling_window(read_micrograph(filename, **extra), (window_size, window_size), (step, step))
    rwin1 = rwin[::2]
    rwin2 = rwin[1::2]
    if rwin1.shape[0] > rwin2.shape[0]: rwin1 = rwin1[:rwin2.shape[0]]
    elif rwin2.shape[0] > rwin1.shape[0]: rwin2 = rwin2[:rwin1.shape[0]]
    avg  = ndimage_utility.powerspec_avg(rwin.reshape( (rwin.shape[0]*rwin.shape[1],   rwin.shape[2],  rwin.shape[3])),  pad)
    #avg1 = ndimage_utility.powerspec_avg(rwin1.reshape((rwin1.shape[0]*rwin1.shape[1], rwin1.shape[2], rwin1.shape[3])), pad)
    #avg2 = ndimage_utility.powerspec_avg(rwin2.reshape((rwin2.shape[0]*rwin2.shape[1], rwin2.shape[2], rwin2.shape[3])), pad)
    ravg = ndimage_utility.rotavg(avg)
    
    #avg = ndimage_utility.replace_outlier(avg, 5, 0, replace=0)
    #avg1 = ndimage_utility.replace_outlier(avg1, 5, 0, replace=0)
    #avg2 = ndimage_utility.replace_outlier(avg2, 5, 0, replace=0)
    mask = eman2_utility.model_circle(10, avg.shape[0], avg.shape[1])*-1+1
    
    write_powerspectra(output, avg*mask, prefix='pow_')
    #write_powerspectra(output, avg1*mask, prefix='pow1_')
    #write_powerspectra(output, avg2*mask, prefix='pow2_')
    
    
    #ravg1 = ndimage_utility.rotavg(avg1)
    #ravg2 = ndimage_utility.rotavg(avg2)
    write_powerspectra(output, ravg*mask, prefix='powavg1_')
    #write_powerspectra(output, ravg1*mask, prefix='powavg1_')
    #write_powerspectra(output, ravg2*mask, prefix='powavg2_')
    
    
    #res = eman2_utility.fsc(avg1, avg2, complex=True)
    #resolution.plot_fsc(format_utility.add_prefix(output, 'half_'), res[:, 0], res[:, 1], apix)
    
    res = eman2_utility.fsc(ravg, avg, complex=True)
    resolution.plot_fsc(format_utility.add_prefix(output, 'rotavg_'), res[:, 0], res[:, 1], apix)
    
def read_micrograph(filename, bin_factor, **extra):
    ''' Read and preprocess a micrograph
    
    :Parameters:
    
    filename : str
               Input filename for the micrograph
    bin_factor : float
                 Number of times to decimate the micrograph
                 
    :Returns:
    
    img : array
          Micrograph image
    '''
    
    img = ndimage_file.read_image(filename)
    if bin_factor != 1.0 and bin_factor != 0.0:
        img = eman2_utility.decimate(img, bin_factor)
    return img

def read_stack(filename, bin_factor, **extra):
    ''' Iterate over a stack and preprocess the images
    
    :Parameters:
    
    filename : str
               Input filename for the particle stack
    bin_factor : float
                 Number of times to decimate the particle stack
                 
    :Returns:
    
    img : array
          Image from particle stack
    '''
    
    for img in ndimage_file.iter_images(filename):
        if bin_factor != 1.0 and bin_factor != 0.0:
            img = eman2_utility.decimate(img, bin_factor)
        yield img

def esimate_astigmatism(powerspec, ampcont, cs, voltage, **extra):
    '''
    
    Use ellipse as a contraint
    '''
    
    pass

def average_powerspec(powerspec, window_len=3):
    '''
    '''
    
    if powerspec.ndim == 1 or powerspec.shape[1] == 1: return powerspec
    avgpowerspec = ndimage_utility.sum_ring(powerspec)
    
    if 1 == 0:
        b = ndimage_utility.rolling_window(avgpowerspec, window_len)
        avg = b.mean(axis=-1)
        avgpowerspec[1:len(avgpowerspec)-1] -= avg
        avgpowerspec[0] -= avg[0]
        avgpowerspec[len(avgpowerspec)-1] -= avg[len(avg)-1]
    
    return avgpowerspec
    

def esimate_defocus(powerspec, ampcont, cs, voltage, **extra):
    '''
    '''
    
    def errfunc(p, y, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3): return y-ctf_model_err(p,cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3)
    avgpowerspec = average_powerspec(powerspec)
    p0 = [1000.0]
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    sfreq = 0.5 / numpy.arange(1, avgpowerspec.shape[0]+1)
    avgpowerspec=avgpowerspec[1:]
    sfreq=sfreq[1:]
    sfreq4 = sfreq**4
    sfreq2 = sfreq**2
    lam3 = lam**3
    arcsin_ampcont = numpy.arcsin(ampcont)
    return scipy.optimize.leastsq(errfunc,p0,args=(avgpowerspec, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3))[0][0]

def ctf_model(n, defocus, ampcont, cs, voltage, **extra):
    '''
    '''
    
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    sfreq = 0.5 / numpy.arange(1, n+1)
    sfreq4 = sfreq**4
    sfreq2 = sfreq**2
    lam3 = lam**3
    arcsin_ampcont = numpy.arcsin(ampcont)
    return sfreq, ctf_model_err(defocus, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3)

def ctf_model_err(defocus, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3=None):
    '''
    
    b-factor: exp(-(n1 + n2sqrt(f)+n3f+n4f^2))
    '''
    
    if lam3 is None: lam3 = lam*lam*lam
    return numpy.sin( (numpy.pi/2.0*cs*lam3*sfreq4 + numpy.pi*lam*defocus*sfreq2) + arcsin_ampcont)
    
def write_powerspectra(output, powerspec, use_powerspec=False, prefix='pow_', **extra):
    ''' Write a power spectra to an output file
    
    :Parameters:
    
    output : str
             Output filename for CTF image plot
    powerspec : array
                Power spectra
    use_powerspec : bool
                    True if input was a power spectra
    prefix : str
             Output filename prefix
    extra : dict
            Unused keyword arguments
    '''
    
    if use_powerspec: return
    base = os.path.basename(output)
    output = os.path.dirname(output)
    output = os.path.join(output, 'pow')
    if not os.path.exists(output): os.makedirs(output)
    output = os.path.join(output, spider_utility.spider_filename(prefix, base))
    ndimage_file.write_image(output, powerspec)

def powerspectra(filename, use_powerspec=False, window_size=500, overlap=0.5, pad=2, **extra):
    ''' Create an averaged power spectra from a stack of images
    
    :Parameters:
    
    filename : str
               Input filename for the micrograph or stack
    use_powerspec : bool
                    True if input was a power spectra
    window_size : int
                  Size of the window
    overlap : float
              Allowed overlap between windows
    pad : int
          Padding for Fourier transform
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    powerspec : array
                Power spectra
    '''
    
    if use_powerspec: return ndimage_file.read_image(filename)
    if ndimage_file.count_images(filename) > 1:
        return ndimage_utility.powerspec_avg(read_stack(filename), pad)
    else:
        step = max(1, window_size*overlap)
        rwin = ndimage_utility.rolling_window(read_micrograph(filename, **extra), (window_size, window_size), (step, step))
        return ndimage_utility.powerspec_avg(rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3])), pad)

def plot_ctf(output, powerspec, defocus, **extra):
    ''' Write a plot of the CTF model and data to an image file
    
    :Parameters:
    
    output : str
             Output filename for CTF image plot
    powerspec : array
                Power spectra
    defocus : float
              Estimated defocus of the power spectra
    extra : dict
            Unused keyword arguments
    '''
    
    if pylab is None: 
        _logger.info("No pylab")
        return
    base = os.path.basename(output)
    output = os.path.dirname(output)
    output = os.path.join(output, 'plot')
    if not os.path.exists(output): os.makedirs(output)
    output = os.path.join(output, spider_utility.spider_filename('plot_', base))
    
    avgpowerspec = average_powerspec(powerspec)
    x, y = ctf_model(avgpowerspec.shape[0], defocus, **extra)
    pylab.clf()
    print "x:", numpy.min(x), numpy.max(x)
    print "y:", numpy.min(y), numpy.max(y)
    print "d:", numpy.min(avgpowerspec), numpy.max(avgpowerspec)
    fig = pylab.figure()
    ax = fig.add_subplot(211)
    ax.plot(x[0:], numpy.log(y[0:]), 'g-.')
    ax.set_xlim(ax.get_xlim()[::-1])
    ax = fig.add_subplot(212)
    #avgpowerspec -= avgpowerspec.min()
    #avgpowerspec /= avgpowerspec.max()
    ax.plot(x[0:], numpy.log(avgpowerspec[0:]), 'r-.')
    ax.set_xlim(ax.get_xlim()[::-1])
    #pylab.axis([0.0,0.5, 0.0,1.0])
    pylab.xlabel('Normalized Frequency')
    pylab.ylabel('CTF')
    pylab.title('Contrast transfer function (CTF)')
    pylab.savefig(os.path.splitext(output)[0]+".png")

def initialize(files, param):
    # Initialize global parameters for the script
    
    _logger.info("Processing %d files"%len(files))
    param["ctf"] = numpy.zeros((len(files), 5))
    '''
    if mpi_utility.is_root(**param):
        radius, offset, bin_factor, param['mask'] = init_param(**param)
        _logger.info("Pixel radius: %d"%radius)
        _logger.info("Window size: %d"%(offset*2))
        if param['bin_factor'] > 1 and not param['disable_bin']: _logger.info("Decimate micrograph by %d"%param['bin_factor'])
        if param['invert']: _logger.info("Inverting contrast of the micrograph")
    '''

def reduce_all(val, ctf, file_index, file_completed, file_count, output, **extra):
    # Process each input file in the main thread (for multi-threaded code)

    filename, ctf_param = val
    ctf[file_index, :] = ctf_param
    
    '''
    format.write(output, ctf[file_index, :].reshape((1, ctf.shape[1])), default_format=format.spiderdoc, 
                 header="id,defocus,astig_ang,astig_mag,cutoff_freq".split(','), mode='a' if file_completed > 1 else 'w', write_offset=file_completed)
    '''
    
    _logger.info("Finished processing: %s"%(os.path.basename(filename)))
    return filename

def finalize(files, ctf, output, **extra):
    # Finalize global parameters for the script
    
    #ctf
    
    # Tune CTF overall values
    _logger.info("Completed - %d"%len(files))

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "CTF Estimation", "Options to control CTF estimation",  id=__name__)
    group.add_option("",   window_size=500,     help="Window size for micrograph periodogram", gui=dict(minimum=10))
    group.add_option("",   overlap=0.5,         help="Allowed overlap for windows in micrograph periodogram", gui=dict(minimum=0.0, decimals=2, singleStep=0.1))
    group.add_option("",   pad=1,               help="Number of times to pad an image before calculating the power spectrum", gui=dict(minimum=1))
    group.add_option("",   use_powerspec=False, help="Input file is already a power spectra")
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)
    pgroup.add_option_group(group)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.window_size <= 0: raise OptionValueError, "Window size must be greater than zero (--window-size)"
    if options.overlap < 0.0 or options.overlap > 1.0 : raise OptionValueError, "Overlap must be between zero and one (--overlap)"
    if options.pad <= 0: raise OptionValueError, "Padding must be greater than zero (--pad)"
    
    options.window_size = int(options.window_size/options.bin_factor)

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    run_hybrid_program(__name__,
        description = '''Estimate CTF parameters from a micrograph or a stack of images
        
                        http://
                        
                        Example: Stack of particles
                         
                        $ ara-defocus input-stack.spi -o defocus.dat -p params
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()



