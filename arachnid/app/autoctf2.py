''' Automatically determine defocus and select good power spectra

.. Created on Jan 11, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.util import plotting
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.image import ndimage_file, ndimage_utility, eman2_utility, ctf, analysis
from ..core.parallel import mpi_utility
import os, numpy, logging, sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, id_len=0, use_emx=False, **extra):
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
    
    if 1 == 0:
        pow = generate_powerspectra(filename, shift=True, **extra)
        vals = ctf.estimate_defocus_fast(pow.copy(), **extra)
        pow = pow.T.copy()
    else:
        pow = generate_powerspectra(filename, shift=False, **extra)
        vals=ctf.search_model_2d(pow, **extra)
        sys.stdout.flush()
        pow=numpy.fft.fftshift(pow).copy()
        opow = pow.copy()
        pow = pow.T.copy()
    ctf.ctf_2d(pow, *vals, **extra)
    pow[:pow.shape[0]/2, :] = ndimage_utility.histeq(pow[:pow.shape[0]/2, :])
    pow[pow.shape[0]/2:, :] = ndimage_utility.histeq(pow[pow.shape[0]/2:, :])
    ndimage_file.write_image(output, pow.T.copy())
    print vals[0], vals[1], numpy.rad2deg(vals[2])
    vals=list(vals)
    vals[2]=numpy.rad2deg(vals[2])
    
    fig, ax=plotting.draw_image(ndimage_utility.histeq(opow), **extra)
    
    tmp = ndimage_utility.normalize_min_max(pow)
    tmp = ndimage_utility.histeq(tmp)
    if 1 == 0: # model mask
        hpow = pow[:pow.shape[0]/2, :pow.shape[0]/2].copy()
        mask = numpy.ones(pow.shape, dtype=numpy.bool)
        mask[:pow.shape[0]/2, :pow.shape[0]/2]=hpow < analysis.otsu(pow)
        tmp = numpy.ma.array(tmp, mask = mask)
        #tmp = numpy.ma.array(tmp, mask = numpy.logical_not(numpy.logical_or(mask == mask.max(), mask == mask.min())))
    else:
        mask = numpy.ones(pow.shape, dtype=numpy.bool)
        mask[:pow.shape[0]/2, :pow.shape[0]/2]=0
        tmp = numpy.ma.array(tmp, mask = mask)
    ax.imshow(tmp, cmap=plotting.cm.gray, alpha=1.0)
    fig.savefig(os.path.splitext(output)[0]+".png", dpi=extra['dpi'], bbox_inches='tight', pad_inches = 0.0)
    
    if use_emx:
        if vals[0] < vals[1]:
            vals = list(vals)
            vals[0], vals[1] = vals[1], vals[0]
            vals[2] = vals[2]-90
            vals[2] = numpy.mod(vals[2], 180.0)
    
    vals = [id, vals[0], vals[1], vals[2], 0.0]
    print 'Found:', vals
    return filename, vals

def generate_powerspectra(filename, bin_factor, invert, window_size, overlap, pad, offset, rmin, rmax, shift=True, from_power=False, cache_pow=False, pow_cache="", multitaper=False, disable_average=False, trans_file="", frame_beg=0, frame_end=-1, **extra):
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
    
    step = max(1, window_size*overlap)
    if from_power:
        return ndimage_file.read_image(filename)
    elif ndimage_file.count_images(filename) > 1 and disable_average:
        pow = None
        _logger.info("Average power spectra over individual frames")
        n = ndimage_file.count_images(filename)
        for i in xrange(n):
            mic = ndimage_file.read_image(filename, i).astype(numpy.float32)
            if bin_factor > 1: mic = eman2_utility.decimate(mic, bin_factor)
            if invert: ndimage_utility.invert(mic, mic)
            rwin = ndimage_utility.rolling_window(mic[offset:mic.shape[0]-offset, offset:mic.shape[1]-offset], (window_size, window_size), (step, step))
            rwin = rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3]))
            if 1 == 1:
                if pow is None: pow = ndimage_utility.powerspec_avg(rwin, pad, shift)/float(n)
                else: pow += ndimage_utility.powerspec_avg(rwin, pad, shift)/float(n)
            else:
                if pow is None:
                    pow, total = ndimage_utility.powerspec_sum(rwin, pad)
                else:
                    _logger.error("%d -- %f"%(total, pow.sum()))
                    total+= ndimage_utility.powerspec_sum(rwin, pad, pow, total)[1]
        pow=ndimage_utility.powerspec_fin(pow, total, shift)
    else:
        if ndimage_file.count_images(filename) > 1 and not disable_average:
            trans = format.read(trans_file, numeric=True, spiderid=filename) if trans_file != "" else None
            mic = None
            if trans is not None:
                if frame_beg > 0: frame_beg -= 1
                if frame_end == -1: frame_end=len(trans)
                for i in xrange(frame_beg, frame_end):
                    frame = ndimage_file.read_image(filename, i)
                    j = i-frame_beg if len(trans) == (frame_end-frame_beg) else i
                    assert(j>=0)
                    frame = eman2_utility.fshift(frame, trans[j].dx, trans[j].dy)
                    if mic is None: mic = frame
                    else: mic += frame
            else:
                for i in xrange(ndimage_file.count_images(filename)):
                    frame = ndimage_file.read_image(filename, i)
                    if mic is None: mic = frame
                    else: mic += frame
        else: 
            mic = ndimage_file.read_image(filename)

        if multitaper:
            if rmin < rmax: rmin, rmax = rmax, rmin
            _logger.info("Estimating multitaper")
            if window_size > 0:
                n=window_size/2
                c = min(mic.shape)/2
                mic=mic[c-n:c+n, c-n:c+n]
            pow = ndimage_utility.multitaper_power_spectra(mic, int(round(rmax)), True, shift)
            #if window_size > 0:
            #    pow = eman2_utility.decimate(pow, float(pow.shape[0])/window_size)
        else:
            _logger.info("Estimating periodogram")
            pow = ndimage_utility.perdiogram(mic, window_size, pad, overlap, offset, shift)
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
        _logger.info("Defocus Search: %f-%f by %f"%(param['dfmin'],param['dfmax'],param['fstep']))
        _logger.info("Resolution Range: %f-%f"%(param['rmin'],param['rmax']))
        _logger.info("Plotting disabled: %d"%plotting.is_plotting_disabled())
        _logger.info("Microscope parameters")
        _logger.info(" - Voltage: %f"%param['voltage'])
        _logger.info(" - CS: %f"%param['cs'])
        _logger.info(" - AmpCont: %f"%param['ampcont'])
        _logger.info(" - Pixel size: %f"%param['apix'])
        #......Add Frame #029 with xy shift:   0.0000   0.0000
        if param['select'] != "":
            select = format.read(param['select'], numeric=True)
            files = spider_utility.select_subset(files, select)
        param['defocus_arr'] = numpy.zeros((len(files), 7))
        try:
            param['defocus_val'] = format_utility.map_object_list(format.read(param['defocus_file'], numeric=True, header=param['defocus_header']))
        except:param['defocus_val']={}
        
        #pow_cache = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_orig', 'pow_'), files[0])
        #if not os.path.exists(os.path.dirname(pow_cache)): os.makedirs(os.path.dirname(pow_cache))
        #param['pow_cache'] = pow_cache
        
        pow_color = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_color', 'pow_'), files[0])
        if not os.path.exists(os.path.dirname(pow_color)): os.makedirs(os.path.dirname(pow_color))
        param['pow_color'] = pow_color
        
        ctf_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'ctf', 'ctf_'), files[0])
        if not os.path.exists(os.path.dirname(ctf_output)): os.makedirs(os.path.dirname(ctf_output))
        param['ctf_output'] = ctf_output
        
        summary_output = spider_utility.spider_template(os.path.join(os.path.dirname(param['output']), 'pow_sum', 'pow'), files[0])
        if not os.path.exists(os.path.dirname(summary_output)): os.makedirs(os.path.dirname(summary_output))
        param['summary_output'] = summary_output
        #pow_cache = spider_utility.spider_filename(pow_cache, files[0])
        #if param['cache_pow'] and os.path.exists(pow_cache):
        #    _logger.info("Using cached power spectra: %s -- %s"%(pow_cache, pow_color))
    return sorted(files)

def reduce_all(filename, file_completed, defocus_arr, defocus_val, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, vals = filename
    
    if len(vals) > 0:
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
        pgroup.add_option("",   dpi=100,              help="Resolution in dots per inche for color figure images")
        
        group = OptionGroup(parser, "Benchmarking", "Options to control benchmarking",  id=__name__)
        group.add_option("-d", defocus_file="", help="File containing defocus values")
        group.add_option("",   defocus_header="id:0,defocus:1", help="Header for file containing defocus values")
        group.add_option("-g", good_file="",    help="Gold standard benchmark")
        pgroup.add_option_group(group)
        
        parser.change_default(log_level=3)
    
    group = OptionGroup(parser, "CTF Estimation", "Options to control CTF Estimation",  id=__name__)
    group.add_option("", dfmin=5000, help="Minimum defocus value to search")
    group.add_option("", dfmax=70000, help="Maximum defocus value to search")
    group.add_option("", fstep=200, help="Defocus search step size")
    group.add_option("", rmin=50.0, help="Minimum resolution to match")
    group.add_option("", rmax=7.5, help="Maximum resolution to match")

    pgroup.add_option_group(group)
    group = OptionGroup(parser, "Power Spectra Creation", "Options to control power spectra creation",  id=__name__)
    group.add_option("", invert=False, help="Invert the contrast - used for unprocessed CCD micrographs")
    group.add_option("", window_size=256, help="Size of the window for the power spec (pixels)")
    group.add_option("", pad=2.0, help="Number of times to pad the power spec")
    group.add_option("", overlap=1.0, help="Amount of overlap between windows")
    group.add_option("", offset=0, help="Offset from the edge of the micrograph (pixels)")
    group.add_option("", from_power=False, help="Input is a powerspectra not a micrograph")
    group.add_option("", pre_decimate=0, help="Size of power spectra for brute force search")
    #group.add_option("", cache_pow=False, help="Save 2D power spectra")
    group.add_option("", mask_radius=0,  help="Mask the center of the color power spectra (Resolution in Angstroms)")
    group.add_option("", disable_average=False,  help="Average power spectra not frames")
    group.add_option("", frame_beg=0,              help="Range for the number of frames")
    group.add_option("", frame_end=-1,             help="Range for the number of frames")
    group.add_option("", trans_file="",  help="Translations for individual frames")
    group.add_option("", multitaper=False,  help="Multi-taper PSD estimation")
    
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

