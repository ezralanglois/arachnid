''' Preprocess a volume

This script (`ara-prepvol`) prepares volume for additional processing such as creating an 
initial reference for refinement.

It performs the following preprocessing on the volume:

    - Resize
    - Filtering
    - Padding or trimming window
    
Todo
 - interpolate to volume size
 - interpolate to longest diameter

Notes
=====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file vol_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. vol_00001.spi. Output files just require the number of digits: `--output vol_prep_0000.spi`

Running the Script
==================

.. sourcecode :: sh
        
    $ ara-prepvol emd_1034.map -o reference_1034.dat --resolution 60

Critical Options
================

.. program:: ara-prepvol

.. option:: --volumes <filename1,filename2>, -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of input volume filenames
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <str>, --output <str>
    
    Output filename template for processed volumes with correct number of digits (e.g. sndc_0000.spi)

Preparation Options
===================

.. program:: ara-prepvol

.. option:: --resolution <float>
    
    Low pass filter volume to given resolution using Gaussian function

.. option:: --apix <float>
    
    Scale volume to the given pixel size

.. option:: --window <int>
    
    Trim or pad volume to given window size
    
More Options
============
    
.. option:: --selection-file <str>
    
    Selection file for a subset of micrographs or selection file template for subset of good particles

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment
    
.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Apr 21, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format
from ..core.metadata import spider_utility
from ..core.metadata import spider_params
from ..core.metadata import selection_utility
from ..core.metadata import format_utility
from ..core.image import ndimage_file
#from ..core.image import ndimage_utility
from ..core.image import ndimage_interpolate
from ..core.image import ndimage_filter
import logging
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, apix, resolution, window, id_len=0, diameter=False, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
        filename : str
                   Input filename
        output : str
                 Output filename
        apix : float
               Target pixel size
        resolution : float
                     Low pass filter
        window : int
               New windows size
        id_len : int, optional
                 Maximum length of the ID
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename, id_len)
    header={}
    vol = ndimage_file.read_image(filename, header=header)
    cur_apix = header['apix']
    _logger.debug("Got pixel size: %f"%cur_apix)
    if cur_apix == 0: raise ValueError, "Pixel size not found in volume header!"
    if resolution > 0:
        _logger.debug("Filtering volume")
        vol = ndimage_filter.filter_gaussian_lowpass(vol, cur_apix/resolution, 2)
    if apix > 0:
        _logger.debug("Interpolating volume")
        # todo -- pad to ensure better interpolation
        vol = ndimage_interpolate.resample_fft_fast(vol, apix/cur_apix)
    else: apix=cur_apix
    if window > 0:
        window = int(window)
        if window > vol.shape[0]:
            _logger.debug("Increasing window size")
            vol = ndimage_filter.pad(vol, tuple([window for _ in xrange(vol.ndim)]))
        elif window < vol.shape[0]:
            _logger.debug("Decreasing window size")
            vol = ndimage_filter.depad_image(vol, tuple([window for _ in xrange(vol.ndim)]))
    _logger.debug("Setting pixel size: %f"%apix)
    ndimage_file.write_image(output, vol, header=dict(apix=apix))
    if diameter:
        from ..core.image import measure
        print measure.estimate_diameter(vol, cur_apix)
        print measure.estimate_shape(vol, cur_apix)
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
    if param.get('window', 0) > 0: _logger.info("Window size: %d"%(param.get('window', 0)))
    if param.get('apix', 0) > 0: _logger.info("Pixel size: %f"%(param.get('apix', 0)))
    if param.get('resolution', 0) > 0: _logger.info("Filter: %f angstroms"%(param.get('resolution', 0)))
    return files

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Prepare", "Options to control volume preparation",  id=__name__)
    group.add_option("", resolution=0.0,        help="Low pass filter volume to given resolution using Gaussian function")
    group.add_option("", apix=0.0,              help="Scale volume to the given pixel size")
    group.add_option("", window=0.0,            help="Trim or pad volume to given window size")
    group.add_option("", diameter=False,        help="Measure diameter of object")
    #group.add_option("", center=('None', 'Mass'),          help="Center volume using specified algorithm")
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", "--volumes", input_files=[], help="List of filenames for the input volumes", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",          help="Output filename template for processed volumes", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", selection_file="",  help="Selection file for a subset of micrographs", gui=dict(filetype="open"), required_file=False)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if (options.window%2) != 0:
        _logger.warn("Window size is odd - this will not work with Relion!")
    
    if len(options.input_files) > 1:
        for filename in options.input_files:
            if not spider_utility.is_spider_filename(filename):
                raise OptionValueError, "Multiple input files must follow SPIDER naming convention and end with a number before the extension, e.g. filename_00001.ext"
        if not spider_utility.is_spider_filename(options.output):
            raise OptionValueError, "When using multiple input files, --output must follow SPIDER naming convention and end with a number before the extension, e.g. filename_00001.ext"

def flags():
    ''' Get flags the define the supported features
    
    :Returns:
    
    flags : dict
            Supported features
    '''
    
    return dict(description = '''Prepare a set of volumes for further processing
                         
                        Example: Run from the command line on a single node
                        
                        $ %prog emd_1034.map -o reference_1034.dat --resolution 60
                      ''',
                supports_MPI=False, 
                supports_OMP=True,
                use_version=True)

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__)

def dependents(): return [spider_params]
if __name__ == "__main__": main()



