''' Estimate the defocus of a micrograph or a stack of particles

This |spi| batch file (`spi-defocus`) estimates the defocus of a set of micrographs or particle stacks.

Tips
====

 #. The micrographs do not have to be in SPIDER format (automatic conversion is attempted). However, 
    :option:`--data-ext` must be used to set the appropriate SPIDER extension.
 
 #. The `bin-factor` options controls the size of multiple parameters, see documentation below for more
    information.
    
Examples
========

.. sourcecode :: sh
    
    # Estimate the defocus over a set of micrographs
    
    $ spi-defocus mic_*.ter -p params.ter -o defocus.ter
    
    # Estimate the defocus over a set of micrographs in TIFF format
    
    $ spi-defocus mic_*.tif -p params.ter -o defocus.ter --data-ext ter
    
    # Esimtate the defocus over a set of image stacks
    
    $ spi-defocus stack_*.ter -p params.ter -o defocus.ter
    
    # Esimtate the defocus over a set of power spectra
    
    $ spi-defocus pow_*.ter -p params.ter -o defocus.ter --use-powerspec

Critical Options
================

.. program:: spi-defocus

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing micrographs, window stacks or power spectra.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)
    
.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <float>
    
    Number of times to decimate params file, and parameters: `window-size`, `x-dist, and `x-dist` and optionally the micrograph

Useful Options
==============

.. option:: --window-size <INT> 
    
    Size of the window to be cropped from the micrograph for the power spectra (Default: 500)

.. option:: --pad <INT> 
    
    Number of times to pad window before Fourier transform; if pad > window-size, then size of padded image (0 means none)

.. option:: --disable-bin <BOOL>
    
    Disable micrograph decimation

Periodogram Options
===================

.. option:: --x-overlap <INT> 
    
    Percent overlap in the x-direction (Default: 50)

.. option:: --y-overlap <INT> 
    
    Percent overlap in the y-direction (Default: 50)

.. option:: --x-dist <INT> 
    
    Distance from the edge in the x-direction (Default: 0)

.. option:: --y-dist <INT> 
    
    Distance from the edge in the y-direction (Default: 0)

Advanced Options
================

.. option:: --output-pow <FILENAME> 
    
    Filename for output power spectra (Default: pow/pow_00000)

.. option:: --output-roo <FILENAME> 
    
    Filename for output rotational average (Default: roo/roo_00000)

.. option:: --output-ctf <FILENAME> 
    
    Filename for output CTF curve (Default: ctf/ctf_00000)

.. option:: --use-powerspec <BOOL>
    
    Set True if the input file is a power spectra (Default: False)

.. option:: --du-nstd <LIST>
    
    List of number of standard deviations for dedusting (Default: empty)

.. option:: --du-type <CHOICE>
    
    Type of dedusting: (1) BOTTOM, (2) TOP, (3) BOTH SIDES: 3 (Default: 3)

.. option:: --ps-radius <INT>
    
    Power spectrum mask radius (Default: 225)

.. option:: --ps-outer <INT>
    
    Power spectrum outer mask radius (Default: 0)

.. option:: --inner-radius <INT> 
    
    Inner mask size for power spectra enhancement (Default: 5)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. todo:: create fast version that does all the power spect averaging in numpy

.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import format, spider_params, spider_utility
from ..core.image import ndimage_file
from ..core.parallel import mpi_utility
from ..core.spider import spider
import os, numpy, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, output_pow="pow/pow_00000", output_roo="roo/roo_00000", output_ctf="ctf/ctf_00000", **extra):
    ''' Esimate the defocus of the given micrograph
    
    :Parameters:
    
    filename : str
               Input micrograph, stack or power spectra file
    output : str
             Output defocus file
    output_pow : str
                 Output power spectra file
    output_roo : str
                 Output rotational average file
    output_ctf : str
                 Output CTF model file
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    filename : str
               Current filename
    '''
    
    _logger.debug("Processing: %s"%filename)
    id = spider_utility.spider_id(filename)
    output = spider_utility.spider_filename(output, id)
    output_pow = spider_utility.spider_filename(output_pow, id)
    output_roo = spider_utility.spider_filename(output_roo, id)
    output_ctf = spider_utility.spider_filename(output_ctf, id)
    #spider.throttle_mp(**extra)
    _logger.debug("create power spec")
    power_spec = create_powerspectra(filename, **extra)
    _logger.debug("mask power spec")
    power_spec = mask_power_spec(power_spec, output_pow=output_pow, **extra)
    _logger.debug("rotational average")
    rotational_average(power_spec, output_roo=output_roo, **extra)
    _logger.debug("estimate defocus")
    ang, mag, defocus, overdef, cutoff, unused = extra['spi'].tf_ed(power_spec, outputfile=output_ctf, **extra)
    return filename, numpy.asarray([id, defocus, ang, mag, cutoff])

def rotational_average(power_spec, spi, output_roo, use_2d=True, **extra):
    '''Compute the rotation average of the power spectra and store as an ndarray
    
    :Parameters:
    
    power_spec : str
                Input filename for power spectra
    spi : spider.Session
          Current SPIDER session
    output_roo : str
                 Output file for rotational average
    use_2d : bool
             Assume 2D power spectra, if true
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    vals : ndarray
           Rotational average of power spectra
    '''
    
    window_size, = spi.fi_h(power_spec, ('NSAM', ))
    rot_avg = spi.ro(power_spec)
    spi.de(output_roo)
    spi.li_d(rot_avg, 'R', 1, outputfile=output_roo, use_2d=use_2d)
    ro_arr = numpy.asarray(format.read(spi.replace_ext(output_roo), numeric=True))
    ro_arr[:, 2] = ro_arr[:, 0]
    ro_arr[:, 0] = numpy.arange(1, len(ro_arr)+1)
    ro_arr[:, 1] = ro_arr[:, 0] / float(window_size)
    format.write(spi.replace_ext(output_roo), ro_arr[:, :3], default_format=format.spiderdoc, header="index,spatial_freq,amplitude".split(','))
    return ro_arr[:, :3]

def mask_power_spec(power_spec, spi, ps_radius=225, ps_outer=0, apix=None, output_pow=None, **extra):
    '''Mask the power spectra
    
    :Parameters:
        
    power_spec : spider_var
                 In-core power spectra
    spi : spider.Session
          Current SPIDER session
    ps_radius : int
                Power spectrum mask radius 
    ps_outer : int
               Power spectrum outer mask radius
    apix : float
           Pixel size in angstroms
    output_pow : str, optional
                 Output file for power spectra
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    masked_ps : spider_var
                Masked power spectra
    '''
    
    if ps_radius == 0: return power_spec
    x_size, y_size, z_size = spi.fi_h(power_spec, ('NSAM', 'NROW', 'NSLICE'))
    x_cent, y_cent = (x_size/2+1, y_size/2+1)
    if z_size > 1:
        center = (x_cent, y_cent, z_size/2+1)
        p_val = spi.gp(power_spec, (x_cent+10, 5, 5))
    else: 
        center = (x_cent, y_cent)
        p_val = spi.gp(power_spec, (x_cent+10, 5))
    mask_radius = int((2*float(apix)/ps_radius)*x_size)
    if ps_outer > 0: ps_outer=x_cent-ps_outer
    power_spec = spi.ma(power_spec, (ps_outer, mask_radius), center, background_type='E', background=p_val)
    if output_pow != "": spi.cp(power_spec, output_pow)
    return power_spec

def create_powerspectra(filename, spi, use_powerspec=False, pad=1, du_nstd=[], du_type=3, **extra):
    ''' Calculate the power spectra from the given file
    
    :Parameters:
    
    filename : str
               Input filename for the micrograph
    spi : spider.Session
          Current SPIDER session
    use_powerspec : bool
                    Set True if the input file is a power spectra
    pad : int
          Number of times to pad window before Fourier transform; if pad > window-size, then size of padded image (0 means none)
    du_nstd : list
              List of number of standard deviations for dedusting
    du_type : int
              Type of dedusting: (1) BOTTOM, (2) TOP, (3) BOTH SIDES: 3
    
    :Returns:
    
    power_sec : spider_var
                In-core reference to power spectra image
    '''
    
    if not use_powerspec:
        image_count = ndimage_file.count_images(filename)
        #image_count = spider.count_images(spi, filename)
        for_win = spider.for_image(spi, filename, image_count) \
                  if image_count > 1 else \
                  for_window_in_micrograph(spi, filename, **extra)
        image_count = 0
        swin = None
        tmp = None
        _logger.debug("generating summed power spec")
        for win in for_win:
            if tmp is None:
                window_size, = spi.fi_h(win, ('NSAM', ))
                if pad > 1 and pad < window_size: pad = pad*window_size
            swin, tmp = periodogram(spi, win, swin, tmp, pad, du_nstd, du_type)
            image_count += 1
        _logger.debug("set image count")
        spi['v10'] = image_count
        _logger.debug("generating avg power spec")
        avg2 = spi.ar(swin, "P1/[v10]")
        _logger.debug("generating std power spec")
        return spi.wu(avg2)
    else:
        power_spec = spi.cp(filename)
    return power_spec

def periodogram(spi, win, swin, tmp, pad=1024, du_nstd=[], du_type=3):
    ''' Calculate an cummulative powerspectra image
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    win : spider_var
          Current window to add
    swin : spider_var
           Cummulative powerspectra
    tmp : spider_var
          Temporary memory
    pad : int
          New padded window size
    du_nstd : list
              List of number of standard deviations for dedusting
    du_type : int
              Type of dedusting
              
    :Returns:
    
    swin : spider_var
           Cummulative powerspectra
    tmp : spider_var
          Temporary memory
    '''
    
    for nstd in du_nstd:  spi.du(win, int(nstd), du_type)
    rwin = spi.ra(win)
    tmp = spi.pd(rwin, pad, background='B', outputfile=tmp) if pad > 1 else rwin
    pwin = spi.pw(tmp, outputfile=tmp)
    win2 = spi.sq(pwin, outputfile=tmp)
    swin = spi.ad(win2, swin, outputfile=swin)
    return swin, tmp

def for_window_in_micrograph(spi, filename, window_size=500, x_overlap=50, y_overlap=50, x_dist=0, y_dist=0, bin_factor=None, **extra):
    ''' Window out successive sliding windows along the micrograph
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : str
               Input filename for the micrograph
    window_size : int
                  Size of the window to be cropped from the micrograph for the power spectra (Default: 500)
    x_overlap : int
                Percent overlap in the x-direction (Default: 50)
    y_overlap : int
                Percent overlap in the y-direction (Default: 50)
    x_dist : int
             Distance from the edge in the x-direction (Default: 0)
    y_dist : int
             Distance from the edge in the y-direction (Default: 0)
    bin_factor : int
                 Decimation factor of the micrograph
    extra : dict
            Unused keyword arguments
                     
    :Returns:
    
    window : spider_var
             Incore file containing the current window
    '''
    
    corefile = read_micrograph_to_incore(spi, filename, bin_factor, **extra)
    window_size /= bin_factor
    if not x_dist: x_dist = window_size
    else: x_dist /= bin_factor
    if not y_dist: y_dist = window_size
    else: y_dist /= bin_factor
    x_size, y_size = spi.fi_h(corefile, ('NSAM', 'NROW'))
    x_overlap_norm = 100.0 / (100-x_overlap)
    y_overlap_norm = 100.0 / (100-y_overlap)
    x_steps = int( float(x_overlap_norm) * ( (x_size-2*x_dist)/window_size-1)   )
    y_steps = int( float(y_overlap_norm) * ( (y_size-2*y_dist)/window_size-1)   )
    x_mult = window_size/x_overlap_norm
    y_mult = window_size/y_overlap_norm
    return for_window_in_section(spi, corefile, window_size, x_mult, y_mult, x_dist, y_dist, x_steps, y_steps)

def read_micrograph_to_incore(spi, filename, bin_factor=1.0, disable_bin=False, local_scratch="", **extra):
    ''' Read a micrograph file into core memory
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : str
               Input filename for the micrograph
    bin_factor : int
                 Decimation factor of the micrograph
    disable_bin : bool
                  Disable micrograph decimatation
    local_scratch : str, optional
                    Output filename for local scratch drive
    
    :Returns:
    
    corefile : spider_var
               Image in SPIDER core memory
    '''
    
    if not os.path.exists(filename): filename = spi.replace_ext(filename)
    temp_spider_file = "temp_spider_file"
    if local_scratch != "" and os.path.exists(local_scratch): temp_spider_file = os.path.join(local_scratch, temp_spider_file)
    filename = ndimage_file.copy_to_spider(filename, spi.replace_ext(temp_spider_file))
    if not disable_bin and bin_factor != 1.0 and bin_factor != 0.0:
        w, h = spider.image_size(spi, filename)[:2]
        corefile = spi.ip(filename, (int(w/bin_factor), int(h/bin_factor)))
        #corefile = spi.dc_s(filename, bin_factor, **extra) # Use iterpolation!
    else:
        corefile = spi.cp(filename, **extra)
    if os.path.exists(spi.replace_ext(temp_spider_file)): os.unlink(spi.replace_ext(temp_spider_file))
    return corefile

def for_window_in_section(spi, corefile, window_size, x_mult, y_mult, x_dist, y_dist, x_steps, y_steps):
    ''' Window out successive sliding windows along a section of the micrograph
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    filename : spider_var
               In-core micrograph reference
    window_size : int
                  Size of the window for the power spectra
    x_mult : int
              Number of pixels to move at each step in the x-direction
    y_mult : int
              Number of pixels to move at each step in the y-direction
    x_dist : int
             Distance from the edge in the x-direction
    y_dist : int
             Distance from the edge in the y-direction
    x_steps : int
              Number of steps in the x-direction
    y_steps : int
              Number of steps in the y-direction
                     
    :Returns:
    
    window : spider_var
             Incore file containing the current window
    '''
    
    tmp = None
    _logger.debug("Steps: %d, %d - %d"%(y_steps, x_steps, window_size))
    for y in xrange(y_steps):
        _logger.debug("%d of %d"%(y, y_steps))
        yu = x_mult * y + y_dist
        for x in xrange(x_steps):
            xu = y_mult * x + x_dist
            win = spi.wi(corefile, (window_size,window_size), (xu,yu), outputfile=tmp)
            yield win

def default_path(filename, output):    
    ''' If the filename is not absolute, then append the path of output file
    
    :Parameters:
    
    filename : str
               Filename for secondary output file
    output : str
             Filename primary output file
    
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    if not os.path.isabs(filename) and os.path.commonprefix( (filename, output ) ) == "":
        path = os.path.dirname(output)
        if path != "": filename = os.path.join(path, filename)
    return filename

def initialize(files, param):
    # Initialize global parameters for the script
    
    if param['output_pow'] == "": param['output_pow']=os.path.join("pow", "pow_00000")
    if param['output_roo'] == "": param['output_roo']=os.path.join("roo", "roo_00000")
    if param['output_ctf'] == "": param['output_ctf']=os.path.join("ctf", "ctf_00000")
    param['output_pow'] = default_path(param['output_pow'], param['output'])
    param['output_roo'] = default_path(param['output_roo'], os.path.dirname(param['output_pow']))
    param['output_ctf'] = default_path(param['output_ctf'],  os.path.dirname(param['output_pow']))
    if mpi_utility.is_root(**param):
        try: os.makedirs(os.path.dirname(param['output_pow'])) 
        except: pass
        try: os.makedirs(os.path.dirname(param['output_roo'])) 
        except: pass
        try: os.makedirs(os.path.dirname(param['output_ctf'])) 
        except: pass
    mpi_utility.barrier(**param)
    param['spi'] = spider.open_session(files, **param)
    spider_params.read(param['spi'].replace_ext(param['param_file']), param)
    param['output'] = param['spi'].replace_ext(param['output'])
    if len(files) > 1 and param['worker_count'] > 1: 
        param['spi'].close()
        param['spi'] = None
    
    if mpi_utility.is_root(**param):
        _logger.info("Writing power spectra to %s"%param['output_pow'])
        _logger.info("Writing defocus to %s"%param['output'])
        _logger.info("Bin factor: %f"%param['bin_factor'])
        _logger.info("Padding: %d"%param['pad'])
        _logger.info("Pixel size: %f"%(param['apix']))
        _logger.info("Window size: %d"%(param['window_size']/param['bin_factor']))
        if param['bin_factor'] != 1.0:
            if not param['disable_bin']:
                _logger.info("Interpolate micrograph with %f"%param['bin_factor'])
            else:
                _logger.info("No micrograph interpolation")

def init_process(process_number, input_files, rank=0, **extra):
    # Initialize a child process
    
    rank = mpi_utility.get_size(**extra)*rank + process_number
    param = {}
    param['spi'] = spider.open_session(input_files, rank=rank, **extra)
    return param
    

def reduce_all(filename, file_completed, file_count, output, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    filename, defocus_vals = filename
    format.write(output, defocus_vals.reshape((1, defocus_vals.shape[0])), default_format=format.spiderdoc, 
                 header="id,defocus,astig_ang,astig_mag,cutoff_freq".split(','), mode='a' if file_completed > 1 else 'w', write_offset=file_completed)
    _logger.info("Finished processing: %s - %d of %d"%(os.path.basename(filename), file_completed, file_count))
    return filename

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc, OptionGroup
    
    pgroup.add_option("-i", input_files=[], help="List of input filenames containing micrographs, window stacks or power spectra", required_file=True, gui=dict(filetype="file-list"))
    pgroup.add_option("-o", output="",      help="Output filename for defocus file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
    spider_params.setup_options(parser, pgroup, True)
    
    group = OptionGroup(parser, "Additional", "Options to customize defocus estimation", group_order=0,  id=__name__)
    group.add_option("",   disable_bin=False,                              help="Disable micrograph decimation")
    group.add_option("",   output_pow=os.path.join("pow", "pow_00000"),    help="Filename for output power spectra", gui=dict(filetype="save"))
    group.add_option("",   output_roo=os.path.join("roo", "roo_00000"),    help="Filename for output rotational average", gui=dict(filetype="save"))
    group.add_option("",   output_ctf=os.path.join("ctf", "ctf_00000"),    help="Filename for output CTF curve", gui=dict(filetype="save"))
    group.add_option("",   inner_radius=5,                                 help="Inner mask size for power spectra enhancement")
    pgroup.add_option_group(group)
    
    setup_options_from_doc(parser, create_powerspectra, mask_power_spec, for_window_in_micrograph, group=pgroup)# classes=spider.Session
    if main_option:
        setup_options_from_doc(parser, spider.open_session)
        parser.change_default(thread_count=4, log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    spider_params.check_options(options)
    if main_option:
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"
        for f in options.input_files:
            if not ndimage_file.is_readable(f): 
                raise OptionValueError, "Unrecognized image format for input-file: %s"%f
        if ndimage_file.is_spider_format(options.input_files[0]) and options.data_ext == "":
            raise OptionValueError, "You must set --data-ext when the input file is not in SPIDER format"

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Estimate the defocus of a set of micrographs or particle stacks
                        
                        http://guam/vispider/vispider/manual.html#module-vispider.batch.defocus
                        
                        $ spi-defocus mic_*.ter -p params.ter -o defocus.ter
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = True,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()


