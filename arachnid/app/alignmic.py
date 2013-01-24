''' Micrograph alignment for tilt, defocus, or dose fractionated series

.. Created on Jan 24, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use('Agg')
from ..core.app import program
from ..core.util import plotting
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.image import ndimage_file, ndimage_utility, eman2_utility, ctf #, analysis
from ..core.image.formats import mrc as mrc_file
from ..core.parallel import mpi_utility
import os, numpy, logging #, scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def process(filename, id_len=0, **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    spider_utility.update_spider_files(extra, spider_utility.spider_id(filename, id_len), 'output', 'pow_cache', 'summary_output')
    pow_cache=extra['pow_cache']
    summary_output=extra['summary_output']
    
    pow1 = generate_powerspectra(average_stack(filename, **extra), **extra)
    s1 = score_powerspectra(pow1)
    _logger.info("Initial score: %f"%s1)
    tot = mrc_file.count_images(filename)
    scores = numpy.zeros((tot, 5))
    sum = read_micrograph(filename, 0, **extra)
    pow = generate_powerspectra(sum, **extra)
    s = score_powerspectra(pow)
    ndimage_file.write_image(pow_cache, pow, 0)
    scores[0, :] = (1, 0, 0, 0, s)
    for i in xrange(1, tot):
        img = read_micrograph(filename, i, **extra)
        ref = sum/i
        cc_map = ndimage_utility.cross_correlate(img, ref)
        y,x = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)
        sum+=eman2_utility.fshift(img, x, y)
        
        pow = generate_powerspectra(sum, **extra)
        s = score_powerspectra(pow)
        ndimage_file.write_image(pow_cache, pow, 0)
        _logger.info("%d - %d, %d, %d, %f"%(i+1, x, y, numpy.hypot(x, y), s))
        scores[i, :] = (i+1, x, y, numpy.hypot(x, y), s)
    sum /= tot
    pow2 = generate_powerspectra(sum, **extra)
    ndimage_file.write_image(pow_cache, pow1, tot+1)
    ndimage_file.write_image(pow_cache, pow2, tot+2)
    s2 = score_powerspectra(pow1)
    _logger.info("Final score: %f"%s2)
    plotting.plot_scatter(summary_output, scores[:, 0], "Index", scores[:, 1], "X")
    plotting.plot_scatter(summary_output, scores[:, 0], "Index", scores[:, 2], "Y")
    plotting.plot_scatter(summary_output, scores[:, 0], "Index", scores[:, 3], "Dist")
    plotting.plot_scatter(summary_output, scores[:, 0], "Index", scores[:, 4], "Score")
    return filename, scores

def average_stack(filename, **extra):
    '''
    '''
    
    tot = mrc_file.count_images(filename)
    sum = read_micrograph(filename, 0, **extra)
    for i in xrange(1, tot):
        sum += read_micrograph(filename, 1, **extra)
    sum /= tot
    return sum

def score_powerspectra(pow):
    '''
    '''
    
    raw = ndimage_utility.mean_azimuthal(pow)[:pow.shape[0]/2]
    raw[1:] = raw[:len(raw)-1]
    raw[:2]=0
    roo = ctf.subtract_background(raw, 21)
    beg = ctf.first_zero(roo)
    end = ctf.energy_cutoff(numpy.abs(roo[beg:]))+beg
    if end == 0: end = len(roo)
    return ctf.estimate_circularity(pow, beg, end)

def generate_powerspectra(avg, window_size, overlap, pad, **extra):
    '''
    '''
    
    #window_size /= bin_factor
    overlap_norm = 1.0 / (1.0-overlap)
    step = max(1, window_size*overlap_norm)
    rwin = ndimage_utility.rolling_window(avg, (window_size, window_size), (step, step))
    try:
        rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
    except:
        _logger.error("%d,%d -- %s -- %s"%(window_size,step, str(rwin.shape), str(avg.shape)))
        raise
    return ndimage_utility.powerspec_avg(rwin, pad)
    
def read_micrograph(filename, index, bin_factor, invert, **extra):
    ''' Read an process a micrograph
    '''
    
    mic = mrc_file.read_image(filename, index)
    if bin_factor > 1: mic = eman2_utility.decimate(mic, bin_factor)
    if invert: ndimage_utility.invert(mic, mic)
    return mic
    
def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        _logger.info("Bin-factor: %f"%param['bin_factor'])
        _logger.info("Invert: %d"%param['invert'])
        _logger.info("Window: %d"%param['window_size'])
        _logger.info("Overlap: %d"%(param['overlap']*100))
       
        param['defocus_arr'] = numpy.zeros((len(files), 7))
        try:
            param['defocus_val'] = format_utility.map_object_list(format.read(param['defocus_file'], numeric=True, header=param['defocus_header']))
        except:param['defocus_val']={}
        
        pow_cache = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_orig', 'pow_'), files[0])
        if not os.path.exists(os.path.dirname(pow_cache)): os.makedirs(os.path.dirname(pow_cache))
        param['pow_cache'] = pow_cache

        summary_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_sum', 'pow'), files[0])
        if not os.path.exists(os.path.dirname(summary_output)): os.makedirs(os.path.dirname(summary_output))
        param['summary_output'] = summary_output

    return sorted(files)

def reduce_all(filename, file_completed, defocus_arr, defocus_val, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, vals = filename
    
    if 1 == 0:
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
    
    if len(files) > 3 and 1 == 0:
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 5], "Bench", defocus_arr[:, 2])
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 2], "Error")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 1], "Defocus", defocus_arr[:, 4], "Resolution")
        plotting.plot_scatter(summary_output, defocus_arr[:, 2], "Error", defocus_arr[:, 3], "Rank")
        plotting.plot_scatter(summary_output, defocus_arr[:, 4], "Resolution", defocus_arr[:, 3], "Rank")
        plotting.plot_histogram_cum(summary_output, defocus_arr[:, 3], 'Rank', 'Micrographs')

    #idx = numpy.argsort(defocus_arr[:, 2])[::-1]
    #format.write(output, defocus_arr[idx], prefix="sel_", format=format.spiderdoc, header="id,defocus,error,rank,resolution,defocus_spi,astig".split(','))
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import OptionGroup 
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", select="",      help="Selection file")
        pgroup.add_option("",   dpi=72,         help="Resolution in dots per inche for color figure images")
        parser.change_default(log_level=3)
    
    group = OptionGroup(parser, "Alignment", "Options to control alignment",  id=__name__)
    group.add_option("", invert=False, help="Invert the contrast - used for unprocessed CCD micrographs")    
    group.add_option("", window_size=256, help="Size of the window for the power spec")
    group.add_option("", pad=2.0, help="Number of times to pad the power spec")
    group.add_option("", overlap=0.5, help="Amount of overlap between windows")
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
        description = '''Align a series of micrographs
                        
                        http://
                        
                        $ ara-alignmic mic_*.ter -p params.ter -o align.ter
                        
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


