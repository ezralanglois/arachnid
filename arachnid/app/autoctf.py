''' Automatically determine defocus and select good power spectra

.. Created on Jan 11, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.util import plotting
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.image import ndimage_file, ndimage_utility, eman2_utility, ctf #, analysis
from ..core.parallel import mpi_utility
import os, numpy, logging #, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, pow_color, bg_window, id_len=0, trunc_1D=False, **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    output : str
             Output defocus file
    bg_window
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    id = spider_utility.spider_id(filename, id_len)
    output = spider_utility.spider_filename(output, id)
    pow = generate_powerspectra(filename, **extra)
    raw = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
    raw[1:] = raw[:len(raw)-1]
    raw[:2]=0
    
    #raw = ndimage_utility.mean_azimuthal(pow)[1:pow.shape[0]/2+1]
    roo = ctf.subtract_background(raw, bg_window)
    extra['ctf_output']=spider_utility.spider_filename(extra['ctf_output'], id)
    
    # initial estimate with outlier removal?
    beg = ctf.first_zero(roo, **extra)
    
    # Estimate by drop in error
    end = ctf.energy_cutoff(numpy.abs(roo[beg:]))+beg
    if end == 0: end = len(roo)
    
    res = ctf.resolution(end, len(roo), **extra)
    cir = ctf.estimate_circularity(pow, beg, end)
    freq = numpy.arange(len(roo), dtype=numpy.float)
    
    roo1, beg1 = ctf.factor_correction(roo, beg, end) if 1 == 1 else roo[beg:end]
    roo1=roo1[beg1:]
    old=beg
    beg = beg1+beg
    # correct for noice and bfactor
    #defocus, err = ctf.estimate_defocus_spi(roo[beg:end], freq[beg:end], len(roo), **extra)[:2]
    defocus, err = ctf.estimate_defocus_spi(roo1, freq[beg:end], len(roo), **extra)[:2]
    
    vals = [id, defocus, err, cir, res]
    
    vextra=[]
    try:
        vextra.append(extra['defocus_val'][id].defocus)
    except:vextra.append(-1)
    
    if len(roo1) == 0: roo1 = roo[beg:end]
    if trunc_1D:
        roo2 = roo1
        rng = (beg, end)
    else:
        #roo2 = roo[beg:]
        roo2, beg1 = ctf.factor_correction(roo, old, len(roo))
        rng = (old, len(roo))
    img = color_powerspectra(pow, roo, roo2, freq, "%d - %.2f - %.2f - %.2f - %.2f - %.2f"%tuple(vals+vextra), defocus, rng, **extra)
    pow_color = spider_utility.spider_filename(pow_color, id)
    if hasattr(img, 'ndim'):
        ndimage_file.write_image(pow_color, img)
    else:
        img.savefig(os.path.splitext(pow_color)[0]+".png", dpi=extra['dpi'], bbox_inches='tight', pad_inches = 0.0)
    
    write_ctf(raw, freq, roo, bg_window, **extra)
    '''
    ;dat/dat   14-JAN-2013 AT 17:42:05   local/ctf/ctf_09755.dat
 ; Spa. freq.(1/A), Back. noise, Back. subtr. PS,  Env(f)**2
    1 4   0.00000       293.113       231.443       354.246     
    2 4  3.487723E-04   293.765       230.791       300.414     
    3 4  6.975447E-04   294.401       141.201       254.358     
    4 4  1.046317E-03   295.018       214.941       214.941     
    5 4  1.395089E-03   295.618       148.703       181.231
    ''' 
    
    return filename, vals

def write_ctf(raw, freq, roo, bg_window, ctf_output, apix, **extra):
    ''' Write a CTF file for CTFMatch
    '''
    
    roo1 = ctf.factor_correction(roo, 0, len(roo))[0]
    bg = ctf.background(raw, bg_window)
    km1 = 1.0 / (2.0*apix)
    dk = km1/len(roo)
    freq = freq*dk
    
    data = numpy.vstack((freq, bg, roo, roo1))
    format.write(ctf_output, data.T, header='freq,roo,bgsub,bgenv'.split(','))

#bg_window

def color_powerspectra(pow, roo, roo1, freq, label, defocus, rng, **extra):
    '''Create enhanced 2D power spectra
    '''
    
    if plotting.is_plotting_disabled(): return pow
    pow = ndimage_utility.replace_outlier(pow, 3, 3, replace='mean')
    fig, ax=plotting.draw_image(pow, label=label, **extra)
    newax = ax.twinx()
    
    model = ctf.ctf_model_spi(defocus, freq, len(roo), **extra)**2
    
    if True:
        linestyle = '-'
        linecolor = 'r'
    else:
        linestyle = ':'
        linecolor = 'w'
        
    freq = freq[rng[0]:]
    model = model[rng[0]:]
    model -= model.min()
    model /= model.max()
    roo1 -= roo1.min()
    roo1 /= roo1.max()
    newax.plot(freq[:rng[1]-rng[0]]+len(roo), model[:rng[1]-rng[0]], c=linecolor, linestyle=linestyle)
    newax.plot(freq[:len(roo1)]+len(roo), roo1, c='w')
    err = numpy.abs(roo1-model[:rng[1]-rng[0]])
    err = ctf.subtract_background(err, 27)
    newax.plot(freq[:rng[1]-rng[0]]+len(roo), err+1.1, c='y')
    
    res = numpy.asarray([ctf.resolution(f, len(roo), **extra) for f in freq[:len(roo1)]])
    
    for v in [6, 8, 12]: #[4, 6, 8, 12]:
        i = numpy.argmin(numpy.abs(v-res))
        newax.text(freq[i]+len(roo), -0.5, "%.1f"%res[i], color='black', backgroundcolor='white', fontsize=4)
        newax.plot([freq[i]+len(roo), freq[i]+len(roo)], [-0.5, 0], ls='-', c='y')
    
    
    val = numpy.max(numpy.abs(roo1))*4
    newax.set_ylim(-val, val)
    newax.set_axis_off()
    newax.get_yaxis().tick_left()
    newax.axes.get_yaxis().set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    newax.get_yaxis().tick_left()
    newax.axes.get_yaxis().set_visible(False)
    #ax.set_xlim(-pow.shape[0]/2, pow.shape[0]/2)
    #ax.set_xlim(0, pow.shape[0])
    #print ax.get_xlim()
    #plotting.trim()
    return fig
    #return plotting.save_as_image(fig)

def generate_powerspectra(filename, bin_factor, invert, window_size, overlap, pad, offset, from_power=False, cache_pow=False, pow_cache="", **extra):
    ''' Generate a power spectra using a perdiogram
    
    :Parameters:
    
    filename : str
               Input filename
    bin_factor : float
                Decimation factor
    invert : bool
             Invert the contrast
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
    
    pow_cache = spider_utility.spider_filename(pow_cache, filename)
    if cache_pow:
        if os.path.exists(pow_cache):
            return ndimage_file.read_image(pow_cache)
    
    mic = ndimage_file.read_image(filename)
    if from_power: return mic
    if bin_factor > 1: mic = eman2_utility.decimate(mic, bin_factor)
    if invert: ndimage_utility.invert(mic, mic)
    #window_size /= bin_factor
    #overlap_norm = 1.0 / (1.0-overlap)
    step = max(1, window_size*overlap)
    rwin = ndimage_utility.rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
    rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
    
    # select good windows
    
    pow = ndimage_utility.powerspec_avg(rwin, pad)
    if cache_pow:
        ndimage_file.write_image(pow_cache, pow)
        plotting.draw_image(ndimage_utility.dct_avg(mic, pad), cmap=plotting.cm.hsv, output_filename=format_utility.add_prefix(pow_cache, "dct_"))
        #pylab.imshow(ccp, cm.jet)
        #ndimage_file.write_image(, )
    return pow
    
def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        _logger.info("Input: %s"%( "Micrograph" if not param['from_power'] else "Power spec"  ))
        #_logger.info("Output: %s"%( "Color" if param['color'] else "Standard"  ))
        _logger.info("Bin-factor: %f"%param['bin_factor'])
        _logger.info("Invert: %f"%param['invert'])
        _logger.info("Window size: %f"%param['window_size'])
        _logger.info("Pad: %f"%param['pad'])
        _logger.info("Overlap: %f"%param['overlap'])
        _logger.info("Plotting disabled: %d"%plotting.is_plotting_disabled())
        if param['select'] != "":
            select = format.read(param['select'], numeric=True)
            files = spider_utility.select_subset(files, select)
        param['defocus_arr'] = numpy.zeros((len(files), 7))
        try:
            param['defocus_val'] = format_utility.map_object_list(format.read(param['defocus_file'], numeric=True, header=param['defocus_header']))
        except:param['defocus_val']={}
        
        pow_cache = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_orig', 'pow_'), files[0])
        if not os.path.exists(os.path.dirname(pow_cache)): os.makedirs(os.path.dirname(pow_cache))
        param['pow_cache'] = pow_cache
        
        pow_color = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_color', 'pow_'), files[0])
        if not os.path.exists(os.path.dirname(pow_color)): os.makedirs(os.path.dirname(pow_color))
        param['pow_color'] = pow_color
        
        ctf_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'ctf', 'ctf_'), files[0])
        if not os.path.exists(os.path.dirname(ctf_output)): os.makedirs(os.path.dirname(ctf_output))
        param['ctf_output'] = ctf_output
        
        summary_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_sum', 'pow'), files[0])
        if not os.path.exists(os.path.dirname(summary_output)): os.makedirs(os.path.dirname(summary_output))
        param['summary_output'] = summary_output
        pow_cache = spider_utility.spider_filename(pow_cache, files[0])
        if param['cache_pow'] and os.path.exists(pow_cache):
            _logger.info("Using cached power spectra: %s -- %s"%(pow_cache, pow_color))
    return sorted(files)

def reduce_all(filename, file_completed, defocus_arr, defocus_val, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, vals = filename
    
    defocus_arr[file_completed-1, :5]=vals
    try:
        defocus_arr[file_completed-1, 5] = defocus_val[int(vals[0])].defocus
    except:
        defocus_arr[file_completed-1, 5]=0
        defocus_arr[file_completed-1, 6]=0
    else:
        try:
            defocus_arr[file_completed-1, 6] = defocus_val[int(vals[0])].astig_mag
        except:
            defocus_arr[file_completed-1, 6] = 0
    
    return filename

def finalize(files, defocus_arr, output, summary_output, good_file="", **extra):
    # Finalize global parameters for the script
    
    if len(files) > 3:
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 5], "Bench", defocus_arr[:, 2])
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 2], "Error")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 4], "Resolution")
        plotting.plot_scatter(summary_output, defocus_arr[:, 2], "Error", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 4], "Resolution", defocus_arr[:, 3], "Rank")
        plotting.plot_histogram_cum(summary_output, defocus_arr[:, 3], 'Rank', 'Micrographs')

    idx = numpy.argsort(defocus_arr[:, 2])[::-1]
    format.write(output, defocus_arr[idx], prefix="sel_", format=format.spiderdoc, header="id,defocus,error,rank,resolution,defocus_spi,astig".split(','))
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup 
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        
        pgroup.add_option("",   disable_color=False, help="Disable output of color 2d power spectra")
        pgroup.add_option("-s", select="",           help="Selection file")
        pgroup.add_option("",   dpi=72,              help="Resolution in dots per inche for color figure images")
        
        group = OptionGroup(parser, "Benchmarking", "Options to control benchmarking",  id=__name__)
        group.add_option("-d", defocus_file="", help="File containing defocus values")
        group.add_option("",   defocus_header="id:0,defocus:1", help="Header for file containing defocus values")
        group.add_option("-g", good_file="",    help="Gold standard benchmark")
        pgroup.add_option_group(group)
        
        parser.change_default(log_level=3)
    
    group = OptionGroup(parser, "Power Spectra Creation", "Options to control power spectra creation",  id=__name__)
    group.add_option("", invert=False, help="Invert the contrast - used for unprocessed CCD micrographs")
    group.add_option("", window_size=256, help="Size of the window for the power spec")
    group.add_option("", pad=2.0, help="Number of times to pad the power spec")
    group.add_option("", overlap=0.5, help="Amount of overlap between windows")
    group.add_option("", offset=0, help="Offset from the edge of the micrograph")
    group.add_option("", from_power=False, help="Input is a powerspectra not a micrograph")
    group.add_option("", bg_window=21, help="Size of the background subtraction window")
    group.add_option("", cache_pow=False, help="Save 2D power spectra")
    group.add_option("", trunc_1D=False, help="Place trunacted 1D/model on color 2D power spectra")
    pgroup.add_option_group(group)
    
def setup_main_options(parser, group):
    # Setup options for the main program
    
    parser.change_default(bin_factor=2)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if options.param_file == "": raise OptionValueError('SPIDER Params file empty')

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Determine defocus and select good power spectra
                        
                        http://
                        
                        $ ara-autoctf mic_*.ter -p params.ter -o defocus.ter
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()

