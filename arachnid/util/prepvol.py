''' Preprocess a volume

This script (`ara-prepvol`) prepares volume for additional processing such as creating an 
initial reference for refinement.

It performs the following preprocessing on the volume:

    - Resize
    - Filtering
    - Padding or trimming window
    - Masking: Tight, sphereical or custom from file
    
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
    
.. option:: --weight <float>
    
    Weight for total variance denosing
    
Mask Options
============

.. option:: --mask-type <None|Adaptive|Sphere|File>
    
    Type of masking

.. option:: --mask-file <FILENAME>
    
    Input filename for existing mask

.. option:: --threshold <FLOAT or STR>
    
    Threshold for density or 'A' for auto threshold - Adaptive

.. option:: --disable-filter
    
    Disable pre filtering - Adaptive

.. option:: --ndilate <INT>
    
    Number of times to dilate the mask - Adaptive

.. option:: --sm-size <INT>
    
    Size of the real space Gaussian kernel (must be odd!) - Adaptive or Sphere - set to 0 to disable

.. option:: --sm-sigma <FLOAT>
    
    Width of the real space Gaussian kernel - Adaptive or Sphere

.. option:: --sphere-pad <INT>
    
    Additional padding on radius of sphere for Sphere mask - Sphere

.. option:: --sphere-radius <FLOAT>
    
    Radius of sphereical mask in angstroms or 'A' for auto measure - Sphere
    
More Options
============
    
.. option:: --selection-file <str>
    
    Selection file for a subset of micrographs or selection file template for subset of good particles

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment
    
.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified

..option:: --diameter
    
    Measure diameter of object

..option:: --cur-apix <FLOAT>
    
    Current pixel size of input volume (only required if not in header)

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
from ..core.image import ndimage_utility
from ..core.image import ndimage_interpolate
from ..core.image import ndimage_filter
try:
    from skimage.filter import denoise_tv_chambolle as tv_denoise  #@UnresolvedImport
    tv_denoise;
except:
    from skimage.filter import tv_denoise  #@UnresolvedImport
import logging
import numpy
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, apix, resolution, window, id_len=0, diameter=False, cur_apix=0, mask_type='None', weight=0, **extra):
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
        diameter : bool
                   Meaure diameter of object
        cur_apix : float
                   Pixel size of input volume
        mask_type : choice
                    Type of masking to perform
        weight : float
                 Regularization parameter for total variance denoising
        extra : dict
                Unused key word arguments
                
    :Returns:
        
        filename : str
                   Current filename
    '''
    
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename, id_len)
    header={}
    vol = ndimage_file.read_image(filename, header=header, force_volume=True)
    if vol.min() == vol.max(): raise ValueError, "Input image has no information: %s"%str(vol.shape)
    if cur_apix == 0: cur_apix = header['apix']
    _logger.debug("Got pixel size: %f"%cur_apix)
    if cur_apix == 0: raise ValueError, "Pixel size not found in volume header! Use --cur-apix to set current pixel size"
    if resolution > 0:
        _logger.debug("Filtering volume")
        vol = ndimage_filter.filter_gaussian_lowpass(vol, cur_apix/resolution, 2)
    if apix > 0:
        _logger.debug("Interpolating volume")
        # todo -- pad to ensure better interpolation
        vol = ndimage_interpolate.resample_fft_fast(vol, apix/cur_apix)
    else: apix=cur_apix
    if weight > 0:
        vol = tv_denoise(vol, weight=weight, eps=2.e-4, n_iter_max=200)
    if window > 0:
        window = int(window)
        if window > vol.shape[0]:
            _logger.debug("Increasing window size")
            vol = ndimage_filter.pad(vol, tuple([window for _ in xrange(vol.ndim)]))
        elif window < vol.shape[0]:
            _logger.debug("Decreasing window size")
            vol = ndimage_filter.depad_image(vol, tuple([window for _ in xrange(vol.ndim)]))
    _logger.debug("Setting pixel size: %f"%apix)
    if mask_type != 'None':
        if mask_type == 'Adaptive':
            mask = tight_mask(vol, **extra)
        elif mask_type == 'Sphere':
            mask = sphere_mask(vol, apix, **extra)
        else:
            mask = ndimage_file.read_image(extra['mask_file'])
        ndimage_file.write_image(format_utility.add_suffix(output, "_mask"), mask, header=dict(apix=apix))
        vol *= mask
    ndimage_file.write_image(output, vol, header=dict(apix=apix))
    if diameter:
        from ..core.image import measure
        print measure.estimate_diameter(vol, cur_apix)
        print measure.estimate_shape(vol, cur_apix)
    return filename

def sphere_mask(vol, apix, sphere_radius, sphere_pad=3, sm_size=3, sm_sigma=3.0, **extra):
    ''' Generate a tight mask for the given volume
        
    :Parameters:
    
        vol : array
              Input volume
        apix : float
               Pixel size
        sphere_radius : int, optional
                        Radius of the sphere
        sphere_pad : int
                     Additional padding on radius of sphere for mask
        sm_size : int
                  Size of the real space Gaussian kernel (must be odd!)
        sm_sigma : float
                   Width of the real space Gaussian kernel
        extra : dict
                Unused key word arguments
    :Returns:
    
        mask : array
               Tight mask
    '''
    
    try: sphere_radius=int(float(sphere_radius)/apix)
    except: 
        from ..core.image import measure
        sphere_radius=int(measure.estimate_diameter(vol, 1.0))
        _logger.info("Estimated radius to be %d"%sphere_radius)
    
    mask = ndimage_utility.model_ball(vol.shape, sphere_radius+sphere_pad, dtype=numpy.float)
    if sm_size > 0 and sm_sigma > 0:
        if (sm_size%2) == 0: sm_size += 1
        ndimage_utility.gaussian_smooth(mask, sm_size, sm_sigma, mask)
    return mask

def tight_mask(vol, threshold=None, ndilate=0, sm_size=3, sm_sigma=3.0, disable_filter=False, **extra):
    ''' Generate a tight mask for the given volume
        
    :Parameters:
    
        vol : array
              Input volume
        threshold : float, optional
                    Threshold for density or None for auto threshold
        ndilate : int
                  Number of times to dilate the mask
        sm_size : int
                  Size of the real space Gaussian kernel (must be odd!)
        sm_sigma : float
                   Width of the real space Gaussian kernel
        disable_prefilter : bool
                            Disable pre filtering
        extra : dict
                Unused key word arguments
    :Returns:
    
        mask : array
               Tight mask
    '''
    
    if sm_size > 0 and (sm_size%2) == 0: sm_size += 1
    try: threshold=float(threshold)
    except: threshold=None
    if not disable_filter:
        fvol = tv_denoise(vol, weight=10, eps=2.e-4, n_iter_max=200)
    else: fvol = vol
    mask, th = ndimage_utility.tight_mask(fvol, threshold, ndilate, sm_size, sm_sigma)
    _logger.info("Determined threshold=%f"%th)
    return mask

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
    if param.get("mask_type", 'None') != 'None':
        if param["mask_type"] == 'Adaptive':
            _logger.info("Using adaptive tight mask with:")
            try: float(param["threshold"])
            except: _logger.info("    - Determine threshold automatically")
            else:   _logger.info("    - Threshold = %f"%float(param["threshold"]))
            if param["ndilate"] > 0: _logger.info("    - Dilation: %d"%param["ndilate"])
            else: _logger.info("    - Dilation disabled")
        if param["mask_type"] == 'Sphere':
            _logger.info("Using sphereical mask with:")
            try: float(param["sphere_radius"])
            except: _logger.info("    - Determine radius automatically")
            else:   _logger.info("    - Radius = %d A"%float(param["sphere_radius"]))
            _logger.info("    - Radius padding: %d"%param["sphere_pad"])
        
        if param["mask_type"] == 'Adaptive' or param["mask_type"] == 'Sphere':
            if param["sm_size"] > 0 and param["sm_sigma"] > 0: _logger.info("    - Soften: %d, %f"%(param["sm_size"], param["sm_sigma"]))
            else: _logger.info("    - Hard edge mask")
        if param["mask_type"] == 'File':
            _logger.info("Using mask from file: %s"%param['mask_file'])
    return files

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Prepare", "Options to control volume preparation",  id=__name__)
    group.add_option("", resolution=0.0,        help="Low pass filter volume to given resolution using Gaussian function")
    group.add_option("", apix=0.0,              help="Scale volume to the given pixel size")
    group.add_option("", cur_apix=0.0,          help="Current pixel size of input volume (only required if not in header)")
    group.add_option("", window=0.0,            help="Trim or pad volume to given window size")
    group.add_option("", weight=0.0,            help="Weight for total variance denosing")
    group.add_option("", diameter=False,        help="Measure diameter of object")
    #group.add_option("", center=('None', 'Mass'),          help="Center volume using specified algorithm")
    mgroup = OptionGroup(parser, "Mask", "Options to control volume masking",  id=__name__)
    mgroup.add_option("", mask_type=('None', 'Adaptive', 'Sphere', 'File'), help="Type of masking")
    mgroup.add_option("", mask_file="",  help="Input filename for existing mask", gui=dict(filetype="open"), required_file=False)
    
    mgroup.add_option("", threshold="A", help="Threshold for density or 'A' for auto threshold - Adaptive")
    mgroup.add_option("", ndilate=0, help="Number of times to dilate the mask - Adaptive")
    mgroup.add_option("", disable_filter=False, help="Disable pre filtering - Adaptive")
    mgroup.add_option("", sm_size=3, help="Size of the real space Gaussian kernel (must be odd!) - Adaptive or Sphere - set to 0 to disable")
    mgroup.add_option("", sm_sigma=3.0, help="Width of the real space Gaussian kernel - Adaptive or Sphere")
    mgroup.add_option("", sphere_pad=3, help="Additional padding on radius of sphere for Sphere mask - Sphere")
    mgroup.add_option("", sphere_radius="A", help="Radius of sphereical mask in angstroms or 'A' for auto measure - Sphere")
    
    group.add_option_group(mgroup)
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



