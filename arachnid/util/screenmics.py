''' Preprocess a set of micrographs for screening

This script (`ara-screenmics`) prepares micrograph image files for screening.

It performs the following preprocessing on the micrograph:

    - High pass filter with cutoff = sigma / window_size (Default sigma = 1, if sigma = 0, then high pass filtering is disabled)
    - Decimation (Default bin-factor = 0, no decimation)
    - Contrast inversion (Default invert = False)
    - Removing bad pixels from dust or errant electrons (Default clamp-window 5, if clamp-window = 0 then disable clamping)

Notes
=====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file mic_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`

 #. Film micrographs - Use the `--film` parameter to disable contrast invertion (if it was done during scanning)
 
Running the Script
==================

.. sourcecode :: sh
        
    $ ara-screenmics mic_*.dat -o mic_sm_00000.dat -p params.dat
    
    # For very compressed images
    
    $ ara-screenmics mic_*.dat -o mic_sm_00000.dat --use-8bit -p params.dat


Critical Options
================

.. program:: ara-screenmics

.. option:: --micrograph-files <filename1,filename2>, -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of input micrograph filenames
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <str>, --output <str>
    
    Output filename template for processed micrograph with correct number of digits (e.g. sndc_0000.spi)

Enhancement Options
===================

.. program:: ara-screenmics

.. option:: --disable-enhance
    
    Disable post-enhancement: clamping, high-pass filtering, decimating

.. option:: --clamp <float>
    
    Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)

.. option:: --sigma <float>
    
    Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)
    
More Options
============
    
.. option:: --selection-file <str>
    
    Selection file for a subset of micrographs or selection file template for subset of good particles

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment
    
.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified
    
.. option:: --film
    
    Do not invert the contrast on the micrograph (usually for film micrographs where inversion was done during scanning)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Feb 14, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format
from ..core.metadata import spider_utility
from ..core.metadata import spider_params
from ..core.metadata import selection_utility
from ..core.metadata import format_utility
from ..core.image import ndimage_file
from ..core.image import ndimage_utility
from ..core.image import ndimage_interpolate
from ..core.image import ndimage_filter
import logging
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, bin_factor, sigma, film, clamp, window=0, disable_enhance=False, use_8bit=False, id_len=0, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
        filename : str
                   Input filename
        output : str
                 Output filename
        bin_factor : float
                     Decimation factor
        sigma : float
                High-pass filter factor
        film : bool
               If True, do not invert contrast
        clamp : float
                       Number of standard deviations from the mean
        window : int
                 Window size of the particle
        disable_enhance : bool
                          Do not performace preprocessing on image
        use_8bit : bool
                   If True, write out 8-bit MRC file
        id_len : int, optional
                 Maximum length of the ID
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    output = spider_utility.spider_filename(output, filename, id_len)
    mic = ndimage_file.read_image(filename)
    if not disable_enhance:
        if bin_factor > 1.0:
            mic = ndimage_interpolate.downsample(mic, bin_factor)
        if sigma > 0.0 and window > 0:
            mic = ndimage_filter.gaussian_highpass(mic, sigma/(window), 2)
        if not film: ndimage_utility.invert(mic, mic)
        if clamp > 0:
            ndimage_utility.replace_outlier(mic, clamp, out=mic)
    
    if use_8bit:
        ndimage_file.write_image_8bit(output, mic, header=dict(apix=extra['apix']))
    else:
        ndimage_file.write_image(output, mic, header=dict(apix=extra['apix']))
    return filename

def initialize(files, param):
    # Initialize global parameters for the script
    
    if param['selection_file'] != "":
        selection_file = format_utility.parse_header(param['selection_file'])[0]
        if os.path.exists(selection_file):
            select = format.read(param['selection_file'], numeric=True)
            _logger.info("Using selection file: %s"%param['selection_file'])
            if len(select) > 0:
                files=selection_utility.select_file_subset(files, select, param.get('id_len', 0), len(param['finished']) > 0)
    if not param['disable_enhance']:
        if param.get('window', 0) > 0:
            _logger.info("Window size: %d"%(param.get('window', 0)))
        if param.get('window', 0) == 0:
            _logger.info("High-pass filtering disabled - no params file")
        if param['sigma'] > 0 and param.get('window', 0)>0: _logger.info("High pass filter: %f"%(param['sigma'] / param['window']))
        if param['bin_factor'] > 1: _logger.info("Decimate micrograph by %d"%param['bin_factor'])
        if not param['film']: _logger.info("Inverting contrast of the micrograph")
        if param['clamp'] > 0: _logger.info("Dedust: %f"%param['clamp'])
    return files

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Screen", "Options to control screening images",  id=__name__)
    group.add_option("", disable_enhance=False,    help="Perform no preprocessing on the micrographs")
    group.add_option("", clamp=2.5,         help="Number of standard deviations to replace extreme values using a Gaussian distribution (0 to disable)")
    group.add_option("", sigma=1.0,                help="Highpass factor: 1 or 2 where 1/window size or 2/window size (0 to disable)")
    group.add_option("", film=False,               help="Do not invert the contrast on the micrograph (inversion is generally done during scanning for film)")
    group.add_option("", use_8bit=False,           help="Write out 8-bit files in the MRC format")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--micrograph-files", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", "--small-micrograph-file", output="", help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", selection_file="",  help="Selection file for a subset of micrographs", gui=dict(filetype="open"), required_file=False)

def change_option_defaults(parser):
    #
    parser.change_default(log_level=3, bin_factor=6.0)

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

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Prepare a set of micrographs for screening
                         
                        Example: Run from the command line on a single node
                        
                        $ %prog mic*.mrc -o mic_00000.spi -p params.spi
                      ''',
                supports_MPI=False, 
                supports_OMP=True,
                use_version=True)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

def dependents(): return [spider_params]
if __name__ == "__main__": main()



