''' Generate a reference for alignment

This |spi| batch file (`spi-reference`) generates a properly filtered and scaled reference for alignment.

Tips
====

 #. The volumes do not have to be in SPIDER format (automatic conversion is attempted). However, 
    :option:`--data-ext` must be used to set the appropriate SPIDER extension.
 
 #. For MRC volumes, the pixel size is extracted from the header, otherwise you must specify it with
    `--curr-apix`.

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Create a reference filtered to 30 A from an MRC map
    
    $ spi-reference emd_1076.map -p params.ter -o reference.ter --data-ext ter -r 30
    
Critical Options
================

.. program:: spi-reference

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing volumes to convert to references. 
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for references with correct number of digits (e.g. enhanced_0000.spi)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

.. option:: --resolution <FLOAT>
    
    Resolution to filter the volumes
    
.. option:: --curr_apix <FLOAT>
    
    Current pixel size of the input volume (only necessary if not MRC)

Low-pass Filter Options
=======================

.. option:: --filter-type <INT>

    Type of low-pass filter to use with resolution: [1] Fermi(SP, fermi_temp) [2] Butterworth (SP-bp_pass, SP+bp_stop) [3] Gaussian (SP)

.. option:: --fermi-temp <FLOAT>

    Fall off for Fermi filter (both high pass and low pass)

.. option:: --bw-pass <FLOAT>

    Offset for pass band of the butterworth lowpass filter (sp-bw_pass)

.. option:: --bw-stop <FLOAT>

    Offset for stop band of the butterworth lowpass filter (sp+bw_stop)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import spider_params, spider_utility
from ..core.parallel import mpi_utility
from ..core.image import ndimage_file
from ..core.spider import spider, spider_file
import filter_volume
import logging, os, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, spi, output, resolution, curr_apix=0.0, disable_center=False, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input volume file
    output : str
             Output reference file
    resolution : float
                 Target resolution to filter reference
    curr_apix : float, optional
                Current pixel size of input map
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    _logger.info("Processing: %s"%os.path.basename(filename))
    _logger.info("Finished: %d,%d"%(0,5))
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename)
    header = ndimage_file.read_header(filename)
    if curr_apix == 0: 
        if header['apix'] == 0: raise ValueError, "Pixel size of input volume is unknown - please use `--curr-apix` to set it"
        curr_apix = header['apix']
    _logger.info("Pixel size: %f for %s"%(curr_apix, filename))
    tempfile = mpi_utility.safe_tempfile(spi.replace_ext('tmp_spi_file'))
    try:
        filename = spider_file.copy_to_spider(filename, tempfile)
    except:
        if os.path.dirname(tempfile) == "": raise
        tempfile = mpi_utility.safe_tempfile(spi.replace_ext('tmp_spi_file'), False)
        filename = spider_file.copy_to_spider(filename, tempfile)
    w, h, d = spi.fi_h(filename, ('NSAM', 'NROW', 'NSLICE'))
    if w != h: raise ValueError, "Width does not match height - requires box"
    if w != d: raise ValueError, "Width does not match depth - requires box"
    _logger.info("Finished: %d,%d"%(1,5))
    _logger.debug("Filtering volume")
    if resolution > 0:
        filename = filter_volume.filter_volume_lowpass(filename, spi, extra['apix']/resolution, outputfile=output, **extra)
    if os.path.exists(tempfile): os.unlink(tempfile)
    _logger.info("Finished: %d,%d"%(2,5))
    _logger.debug("Resizing volume")
    filename = resize_volume(filename, spi, curr_apix, outputfile=output, **extra)
    _logger.info("Finished: %d,%d"%(3,5))
    if not disable_center:
        _logger.debug("Centering volume")
        filename = center_volume(filename, spi, output)
    _logger.info("Finished: %d,%d"%(4,5))
    return filename

def center_volume(filename, spi, output):
    ''' Center the volume in the box
    
    :Parameters:
    
    filename : str
               Input volume file
    spi : spider.Session
          Current SPIDER session
    output : str
             Output centered volume file
             
    :Returns:
    
    output : str
             Output centered volume file
    '''
    
    if filename == output: filename = spi.cp(filename)
    coords = spi.cg_ph(filename)
    return spi.sh_f(filename, tuple(-numpy.asarray(coords[3:])), outputfile=output)

def resize_volume(filename, spi, curr_apix, apix, window, outputfile=None, **extra):
    ''' Interpolate the volume and change the box size to match the params file
    
    :Parameters:
    
    filename : str
               Input volume file
    spi : spider.Session
          Current SPIDER session
    curr_apix : float
                Pixel size of input volume
    apix : float
           Target pixel size (params file)
    window : float
             Target window size (params file)
    output : str
             Output interpolated volume file
             
    :Returns:
    
    output : str
             Output interpolated volume file
    '''
    
    w, h, d = spi.fi_h(filename, ('NSAM', 'NROW', 'NSLICE'))
    if w != h: raise ValueError, "Width does not match height - requires box"
    if w != d: raise ValueError, "Width does not match depth - requires box"
    
    if not numpy.allclose(curr_apix, apix):
        bin_factor = curr_apix / apix
        _logger.info("Interpolating Structure: %f * %f = %f | %f/%f | %f"%(w, bin_factor, w*bin_factor, apix, curr_apix, window))
        w *= bin_factor
        h *= bin_factor
        d *= bin_factor
        filename = spi.ip(filename, (int(w), int(h), int(d)))
    
    if w < window:
        _logger.info("Increasing window size from %d -> %d"%(w, window))
        filename = spi.pd(filename, window, outputfile=outputfile)
    elif w > window:
        _logger.info("Decreasing window size from %d -> %d"%(w, window))
        filename = spi.wi(filename, window, outputfile=outputfile)
    return outputfile

def initialize(files, param):
    # Initialize global parameters for the script
    
    if len(files) == 0: return
    if param['param_file'] != "" and os.path.splitext(param['param_file'])[1] != "":
        files=[param['param_file']]
    param['spi'] = spider.open_session(files, **param)
    if param['new_window'] > 0:
        param['bin_factor'] = float(param['window'])/param['new_window']
    spider_params.read(param['spi'].replace_ext(param['param_file']), param)
    
def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input filenames containing volumes to convert to references", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for references with correct number of digits (e.g. enhanced_0000.spi)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, True)
        pgroup.add_option("-r", resolution=30.0,        help="Resolution to filter the volumes")
        pgroup.add_option("",   curr_apix=0.0,          help="Current pixel size of the input volume (only necessary if not MRC)")
        pgroup.add_option("",   new_window=0.0,         help="Set bin_factor based on new window size")
        pgroup.add_option("",   disable_center=False,   help="Disable centering")
        setup_options_from_doc(parser, spider.open_session, group=pgroup)
        parser.change_default(thread_count=4, log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        spider_params.check_options(options)
        #if options.resolution <= 0.0: raise OptionValueError, "--resolution must be a positive value greater than 0"
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Generate a reference for alignment
                        
                        http://guam/vispider/vispider/manual.html#module-vispider.batch.reference
                        
                        $ spi-reference emd_1076.map -p params.ter -o reference.ter --data-ext ter -r 30
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
def dependents(): return [filter_volume]
if __name__ == "__main__": main()

