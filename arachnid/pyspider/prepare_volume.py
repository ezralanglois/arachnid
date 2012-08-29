''' Volume processing for next round of refinement.

This pySPIDER batch file (`spi-prepvol`) takes sets of three volumes (full volume and two half volumes)
produced by alignment/refinement and then does the following:
    
    #. Masks the half volumes
    
    #. Calculates the resolution between the half volumes
    
    #. Filters the full volume
    
    #. Masks the full filtered volume

Tips
====

 #. The input files should be the full volume, followed by the two half volumes.
 
 #. If the raw volumes follow the default |spi| naming scheme (e.g. raw_vol01.spi raw1_vol01.spi raw2_vol01.spi) then
    you may use the following as the input file: raw*_vol01.spi (rather than raw_vol01.spi raw1_vol01.spi raw2_vol01.spi)

Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Calculate the resolution of two half volumes and filter the input raw volume to the resolution
    
    $ spi-prepvol raw_vol01.spi raw1_vol01.spi raw2_vol01.spi -p params.spi -o filt_vol_0001.spi
    
    # Do the same as the first example, but apply a Gaussian mask of width 3 pixels to the output volume
    
    $ spi-prepvol raw_vol01.spi raw1_vol01.spi raw2_vol01.spi -p params.spi -o filt_vol_0001.spi --volume-mask G --mask-edge-width 3
    
    # Do the same as the first example, but apply a user-defined mask to the half volumes before resolution calculation
    
    $ spi-prepvol raw_vol01.spi raw1_vol01.spi raw2_vol01.spi -p params.spi -o filt_vol_0001.spi --resolution-mask user_defined_mask.spi

.. todo:: 
    
    #. Ensure mask files exist
    
    #. Append local root in setup_options to mask files
    
    #. Decimate mask files

Critical Options
================

.. program:: spi-prepvol

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing volumes triples, full_vol, half_vol_1, half_vol_2
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for the filtered, masked volume as well as base output name for FSC curve (`res_$output`)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

.. option:: volume-mask <('A', 'C', 'G' or FILENAME)>
    
    Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask


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

High-pass Filter Options
========================

.. option:: --hp-radius <FLOAT>

    The spatial frequency to high-pass filter (if > 0.5, then assume its resolution and calculate spatial frequency, if 0 the filter is disabled)

.. option:: --hp-type <INT>

    Type of high-pass filter to use with resolution: [0] None [1] Fermi(hp_radius, fermi_temp) [2] Butterworth (hp_radius-bp_pass, hp_radius+bp_stop) [3] Gaussian (hp_radius)

.. option:: --hp-bw_pass <FLOAT>

    Offset for the pass band of the butterworth highpass filter (hp_radius-bw_pass)

.. option:: --hp-bw_stop <FLOAT>

    Offset for the stop band of the butterworth highpass filter (hp_radius+bw_stop)

.. option:: --hp-temp <FLOAT>

    Temperature factor for the fermi filter
    
Spherical Mask Options
=======================

.. option:: --mask-edge-width <INT>

    Set edge with of the mask (for Gaussian this is the half-width)

Adaptive Tight Mask Options
===========================

.. option:: --threshold <STR or FLOAT>

    Threshold for density or 'A' for auto threshold
    
.. option:: --ndilate <INT>

    Number of times to dilate the mask

.. option:: --gk-size <INT>
 
     Size of the real space Gaussian kernel (must be odd!)

.. option:: --gk-sigma <FLOAT>

    Width of the real space Gaussian kernel
    
Resolution Mask Options
=======================

.. option:: --resolution-mask : str

    Set the type of mask: C for cosine and G for Gaussian and N for no mask and A for adaptive tight mask or a filepath for external mask

.. option:: --res-edge-width : int

    Set edge with of the mask (for Gaussian this is the half-width)

.. option:: --res-threshold : str

    Threshold for density or 'A' for auto threshold

.. option:: --res-ndilate : int

    Number of times to dilate the mask

.. option:: --res-gk-size : int

    Size of the real space Gaussian kernel (must be odd!)

.. option:: --res-gk-sigma : float

    Width of the real space Gaussian kernel

Resolution Options
==================

These options are passed to SPIDER's `rf_3` command, the default values are generally fine for 
most experiments.

.. option:: --ring-width <float> 
    
    Shell thickness in reciprocal space sampling units (Default: 0.5)

.. option:: --lower-scale <float> 
    
     Lower range of scale factors by which the second Fourier must be multiplied for the comparison (Default: 0.2)

.. option:: --upper-scale <float> 
    
    Upper range of scale factors by which the second Fourier must be multiplied for the comparison (Default: 2.0)

.. option:: --missing-ang <choice('C' or 'W')> 
    
    'C' if you have a missing cone and 'W' if you have a missing wedge (Default: 'C')

.. option:: --max-tilt <float> 
    
    Angle of maximum tilt angle in degrees (Default: 90.0)

.. option:: --noise-factor <float> 
    
    Factor given here determines the FSCCRIT. Here 3.0 corresponds to the 3 sigma criterion i.e., 3/SQRT(N), 
    where N is number of voxels for a given shell.

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    #. :ref:`Options shared by |spi| scripts... <spider-options>`
    #. :ref:`Options shared by MPI-enabled scripts... <mpi-options>`
    #. :ref:`Options shared by file processor scripts... <file-proc-options>`
    #. :ref:`Options shared by SPIDER params scripts... <param-options>`

.. Created on Jul 15, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.metadata import spider_params, spider_utility, format_utility
from ..core.spider import spider
import filter_volume, resolution, mask_volume, enhance_volume
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input volume file
    output : str
             Output reference file
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename)
    
    res = post_process(filename, output=output, output_volume=format_utility.add_prefix(output, "vol_"), **extra)
    _logger.info("Resolution = %f"%res)
    return filename

def post_process(files, spi, output, output_volume="", min_resolution=0.0, add_resolution=0.0, enhance=False, **extra):
    ''' Postprocess reconstructed volumes for next round of refinement
    
    :Parameters:
    
    filename : str
               Filename of the input volume
    spi : spider.Session
          Current SPIDER session
    output : str
             Output filename base for postprocessed volume
    min_resolution : float
                     Minimum resolution for filtering the structure
    add_resolution : float
                     Additional amount to add to resolution before filtering the next reference
    output_volume : str
                    Output filename for the reconstructed volume (if empty, `vol_$output` will be used). half volumes will be prefixed with `h1_` and `h2_` and the raw volume, `raw_`
    enhance : bool
              Output an enhanced density map
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    resolution : float
                 Current resolution for the reconstruction
    '''
    
    if output_volume == "": output_volume = format_utility.add_prefix(output, "vol_")
    sp = resolution.estimate_resolution(files[1], files[2], spi, format_utility.add_prefix(output, "dres_"), **extra)
    res = extra['apix']/sp
    if add_resolution > 0.0: 
        sp = extra['apix'] / (add_resolution+res)
    if (add_resolution+res) < min_resolution: sp = extra['apix']/min_resolution
    output_volume = filter_volume.filter_volume_highpass(files[0], outputfile=output_volume, **extra)
    output_volume = filter_volume.filter_volume_lowpass(output_volume, sp, outputfile=output_volume, **extra)
    output_volume = mask_volume.mask_volume(output_volume, output_volume, spi, **extra)
    if enhance:
        enhance_volume.enhance_volume(spi, extra['apix'] / res, format_utility.add_prefix(output, "enh_"), **extra)
    return res

def initialize(files, param):
    # Initialize global parameters for the script
    
    param['spi'] = spider.open_session(files, **param)
    spider_params.read_spider_parameters_to_dict(param['spi'].replace_ext(param['param_file']), param)
    pfiles = []
    for i in xrange(0, len(files), 3):
        pfiles.append((files[i], files[i+1], files[i+2]))
    return pfiles

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        parser.add_option("-i", input_files=[], help="List of input filenames containing volumes triples, full_vol, half_vol_1, half_vol_2", required_file=True, gui=dict(filetype="file-list"))
        parser.add_option("-o", output="",      help="Output filename for the filtered, masked volume as well as base output name for FSC curve (`res_$output`)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, True)
    setup_options_from_doc(parser, filter_volume.filter_volume_highpass)
    if main_option:
        parser.change_default(thread_count=4, log_level=3)
    
def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        spider_params.check_options(options)
        if len(options.input_files)%3 != 0:
            _logger.debug("Found: %s"%",".join(options.input_files))
            raise OptionValueError, "Requires input files in sets of 3, e.g. full_vol,half1_vol,half2_vol - found %d"%len(options.input_files)
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Prepare a volume for refinement
                        
                        $ %prog raw_vol01.spi raw1_vol01.spi raw2_vol01.spi -p params.spi -o filt_vol_0001.spi
                        
                        http://guam/vispider/vispider/manual.html#module-vispider.batch.prepare_volume
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=True,
        use_version = False,
        max_filename_len = 78,
    )
def dependents(): return [filter_volume, resolution, mask_volume]
if __name__ == "__main__": main()



