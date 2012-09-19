''' Estimate the CTF parameters from a micrograph or a stack of particles


.. Created on Sep 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import format, spider_params, spider_utility, format_utility
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
    powerspec = powerspectra(filename, **extra)
    write_powerspectra(output, powerspec, **extra)
    defocus = esimate_defocus(powerspec, **extra)
    plot_ctf(output, powerspec, defocus, **extra)
    
    # Calculate astig
    
    return filename, numpy.asarray((id, defocus, 0, 0, 0))


def esimate_signal(filename, output, apix, window=500, overlap=0.5, pad=2, **extra):
    ''' Estimate the signal based on two power spectra
    
    Use ellipse as a contraint
    '''
    
    step = max(1, window*overlap)
    rwin = ndimage_utility.rolling_window(ndimage_file.read_image(filename), (window, window), (step, window))
    rwin1 = rwin[::2]
    rwin2 = rwin[1::2]
    avg1 = ndimage_utility.powerspec_avg(rwin1.reshape((rwin1.shape[0]*rwin1.shape[1], rwin1.shape[2], rwin1.shape[3])), pad)
    avg2 = ndimage_utility.powerspec_avg(rwin2.reshape((rwin2.shape[0]*rwin2.shape[1], rwin2.shape[2], rwin2.shape[3])), pad)
    
    write_powerspectra(output, avg1, prefix='pow1_')
    write_powerspectra(output, avg2, prefix='pow2_')
    
    res = eman2_utility.fsc(avg1, avg2)
    resolution.plot_fsc(format_utility.add_prefix(output, 'raw_'), res[:, 0], res[:, 1], apix)
    
    ravg = ndimage_utility.rotavg(avg1)+ndimage_utility.rotavg(avg2)
    mask = ndimage_utility.segment(ravg)
    
    mavg1 = avg1*mask
    mavg2 = avg2*mask
    write_powerspectra(output, mavg1, prefix='mpow1_')
    write_powerspectra(output, mavg2, prefix='mpow2_')
    
    res = eman2_utility.fsc(mavg1, mavg2)
    resolution.plot_fsc(format_utility.add_prefix(output, 'masked_'), res[:, 0], res[:, 1], apix)

def esimate_astigmatism(powerspec, ampcont, cs, voltage, **extra):
    '''
    
    Use ellipse as a contraint
    '''
    
    pass

def esimate_defocus(powerspec, ampcont, cs, voltage, **extra):
    '''
    '''
    
    def errfunc(p, y, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3): return y-ctf_model_err(p,cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3)
    avgpowerspec = ndimage_utility.mean_azimuthal(powerspec)
    p0 = [1000.0]
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    sfreq = 0.5 / numpy.arange(avgpowerspec.shape[0])
    avgpowerspec=avgpowerspec[1:]
    sfreq=sfreq[1:]
    sfreq4 = sfreq**4
    sfreq2 = sfreq**2
    lam3 = lam**3
    arcsin_ampcont = numpy.arcsin(ampcont)
    return scipy.optimize.leastsq(errfunc,p0,args=(avgpowerspec, cs, lam, sfreq4, sfreq2, arcsin_ampcont, lam3))[0]

def ctf_model(n, defocus, ampcont, cs, voltage, **extra):
    '''
    '''
    
    lam = 12.398 / numpy.sqrt(voltage * (1022.0 + voltage))
    sfreq = 0.5 / numpy.arange(n)
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

def powerspectra(filename, use_powerspec=False, window=500, overlap=0.5, pad=2, **extra):
    ''' Create an averaged power spectra from a stack of images
    
    :Parameters:
    
    filename : str
               Input filename for the stack
    use_powerspec : bool
                    True if input was a power spectra
    window : int
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
        return ndimage_utility.powerspec_avg(ndimage_file.iter_images(filename), pad)
    else:
        step = max(1, window*overlap)
        rwin = ndimage_utility.rolling_window(ndimage_file.read_image(filename), (window, window), (step, step))
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
    
    if pylab is None: return
    base = os.path.basename(output)
    output = os.path.dirname(output)
    output = os.path.join(output, 'plot')
    if not os.path.exists(output): os.makedirs(output)
    output = os.path.join(output, spider_utility.spider_filename('plot_', base))
    
    avgpowerspec = ndimage_utility.mean_azimuthal(powerspec)
    x, y = ctf_model(avgpowerspec.shape[0], defocus, **extra)
    pylab.clf()
    pylab.plot(x, y, 'g.')
    pylab.plot(x, avgpowerspec)
    pylab.axis([0.0,0.5, 0.0,1.0])
    pylab.xlabel('Normalized Frequency')
    pylab.ylabel('CTF')
    pylab.title('Contrast transfer function (CTF)')
    pylab.savefig(os.path.splitext(output)[0]+".png")

def initialize(files, param):
    # Initialize global parameters for the script
    
    param["ctf"] = numpy.zeros((len(files), 4))
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
    
    format.write(output, ctf.reshape((1, ctf.shape[0])), default_format=format.spiderdoc, 
                 header="id,defocus,astig_ang,astig_mag,cutoff_freq".split(','), mode='a' if file_completed > 1 else 'w', write_offset=file_completed)
    
    _logger.info("Finished processing: %s"%(os.path.basename(filename)))
    return filename

def finalize(files, ctf, output, **extra):
    # Finalize global parameters for the script
    
    #ctf
    
    # Tune CTF overall values
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "CTF Estimation", "Options to control CTF estimation",  id=__name__)
    group.add_option("",   window=500,          help="Window size for micrograph periodogram", gui=dict(minimum=10))
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
    if options.window <= 0: raise OptionValueError, "Window size must be greater than zero (--window)"
    if options.overlap < 0.0 or options.overlap > 1.0 : raise OptionValueError, "Overlap must be between zero and one (--overlap)"
    if options.pad <= 0: raise OptionValueError, "Padding must be greater than zero (--pad)"

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



