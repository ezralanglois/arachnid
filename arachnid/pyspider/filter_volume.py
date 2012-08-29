''' Filter a volume

This pySPIDER batch file (`spi-filtervol`) filters a volume to a specific resolution.

Tips
====

 #. Boolean flags such as `disable-enhance` cannot take a value on the command-line. They only require a value in a configuration 
    file. Their existence or absence sets the appropriate value. For example, specifiying 
    `$ spi-filtervol --disable-enhance ....` will disable amplitude enhancement.
    
Examples
========

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Filter two volumes (must have suffix numbers and they must be different) to 15 A
    
    $ spi-filtervol vol1.spi vol2.spi -p params.spi -o filt_vol_0001.spi -r 15
    
    # Filter and enhance a volume
    
    $ spi-filtervol vol1.spi vol2.spi -p params.spi -o filt_vol_0001.spi -r 15 --scatter-doc scattering8.spi

Critical Options
================

.. program:: spi-filtervol

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames containing volumes. 
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output filename for filtered volume with correct number of digits (e.g. masked_0000.spi)

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Path to SPIDER params file

.. option:: --bin-factor <FLOAT>
    
    Number of times to decimate params file

.. option:: --resolution <FLOAT>
    
    Resolution to filter the volumes

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
from ..core.metadata import spider_params, spider_utility
from ..core.spider import spider
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(filename, output, resolution, **extra):
    ''' Create a reference from from a given density map
    
    :Parameters:
    
    filename : str
               Input volume file
    output : str
             Output reference file
    resolution : float
                 Resolution to filter
    extra : dict
            Unused key word arguments
             
    :Returns:
    
    filename : str
               Filename for correct location
    '''
    
    if spider_utility.is_spider_filename(filename):
        output = spider_utility.spider_filename(output, filename)
    sp = extra['apix']/resolution
    output = filter_volume_highpass(filename, outputfile=output, **extra)
    output = filter_volume_lowpass(output, sp, outputfile=output, **extra)
    return filename

def filter_volume_lowpass(filename, spi, sp, filter_type=2, fermi_temp=0.0025, bw_pass=0.05, bw_stop=0.05, outputfile=None, **extra):
    ''' Low-pass filter the specified volume
    
    :Parameters:
    
    filename : str
               Filename of the input volume
    spi : spider.Session
          Current SPIDER session
    sp : float
         Spatial frequency to filter volume
    filter_type : int
                  Type of low-pass filter to use with resolution: [1] Fermi(SP, fermi_temp) [2] Butterworth (SP-bp_pass, SP+bp_stop) [3] Gaussian (SP)
    fermi_temp : float
                 Fall off for Fermi filter (both high pass and low pass)
    bw_pass : float
              Offset for pass band of the butterworth lowpass filter (sp-bw_pass)
    bw_stop : float
              Offset for stop band of the butterworth lowpass filter (sp+bw_stop)
    outputfile : str
                 Output filename for filtered volume
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
                 Output filename for filtered volume
    '''
    
    if sp > 0.08:
        if filter_type == 1:
            rad = sp
            if rad > 0.45: rad = 0.45
            outputfile = spi.fq(filename, spi.FERMI_LP, filter_radius=rad, temperature=fermi_temp, outputfile=outputfile)
        else:
            pass_band = sp-bw_pass
            stop_band = sp+bw_stop
            if pass_band > 0.35: pass_band = 0.4
            if stop_band > 0.4: stop_band = 0.45
            outputfile = spi.fq(filename, spi.BUTER_LP, pass_band=pass_band, stop_band=stop_band, outputfile=outputfile)
    else:
        _logger.warn("Spatial frequency %f exceeds the safe value, switching to Gaussian filter"%(sp))
        filter_type = 3
    if filter_type == 3:
        outputfile = spi.fq(filename, spi.GAUS_LP, filter_radius=sp, outputfile=outputfile)
    return outputfile

def filter_volume_highpass(filename, spi, hp_radius=0, hp_type=0, hp_bw_pass=0.05, hp_bw_stop=0.05, hp_temp=0.0025, apix=None, outputfile=None, **extra):
    ''' High-pass filter the specified volume
    
    :Parameters:
    
    filename : str
               Filename of the input volume
    spi : spider.Session
          Current SPIDER session
    hp_radius : float
                The spatial frequency to high-pass filter (if > 0.5, then assume its resolution and calculate spatial frequency, if 0 the filter is disabled)
    hp_type : int
              Type of high-pass filter to use with resolution: [0] None [1] Fermi(hp_radius, fermi_temp) [2] Butterworth (hp_radius-bp_pass, hp_radius+bp_stop) [3] Gaussian (hp_radius)
    hp_bw_pass : float
                 Offset for the pass band of the butterworth highpass filter (hp_radius-bw_pass)
    hp_bw_stop : float
                 Offset for the stop band of the butterworth highpass filter (hp_radius+bw_stop)
    hp_temp : float
              Temperature factor for the fermi filter
    apix : float
           Pixel size
    outputfile : str
                 Output filename for filtered volume
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    outputfile : str
                 Output filename for filtered volume
    '''
    
    if filename == outputfile: filename = spi.cp(filename)
    if hp_radius > 0.5: hp_radius = apix / hp_radius
    if hp_type == 1:
        outputfile = spi.fq(filename, spi.FERMI_HP, filter_radius=hp_radius, temperature=hp_temp, outputfile=outputfile)
    elif hp_type == 2:
        pass_band = hp_radius-hp_bw_pass
        stop_band = hp_radius+hp_bw_stop
        if pass_band > 0.35: pass_band = 0.4
        if stop_band > 0.4: stop_band = 0.45
        outputfile = spi.fq(filename, spi.BUTER_HP, pass_band=pass_band, stop_band=stop_band, outputfile=outputfile)
    elif hp_type == 3:
        outputfile = spi.fq(filename, spi.GAUS_HP, filter_radius=hp_radius, outputfile=outputfile)
    else: return filename
    return outputfile

def initialize(files, param):
    # Initialize global parameters for the script
    
    param['spi'] = spider.open_session(files, **param)
    spider_params.read_spider_parameters_to_dict(param['spi'].replace_ext(param['param_file']), param)

def finalize(files, **extra):
    # Finalize global parameters for the script
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    from ..core.app.settings import setup_options_from_doc
    
    if main_option:
        parser.add_option("-i", input_files=[], help="List of input filenames containing volumes", required_file=True, gui=dict(filetype="file-list"))
        parser.add_option("-o", output="",      help="Output filename for filtered volume with correct number of digits (e.g. masked_0000.spi)", gui=dict(filetype="save"), required_file=True)
        spider_params.setup_options(parser, pgroup, True)
        parser.add_option("-r", resolution=15.0,        help="Resolution to filter the volumes")
    setup_options_from_doc(parser, filter_volume_lowpass)
    if main_option:
        setup_options_from_doc(parser, spider.open_session, filter_volume_highpass)
        parser.change_default(thread_count=4, log_level=3)
    

def check_options(options, main_option=False):
    #Check if the option values are valid
    from ..core.app.settings import OptionValueError
    
    if main_option:
        spider_params.check_options(options)
        if options.resolution <= 0.0: raise OptionValueError, "--resolution must be a positive value greater than 0"
        if not spider_utility.test_valid_spider_input(options.input_files):
            raise OptionValueError, "Multiple input files must have numeric suffix, e.g. vol0001.spi"

def main():
    #Main entry point for this script
    from ..core.app.program import run_hybrid_program
    
    run_hybrid_program(__name__,
        description = '''Filter a volume(s)
                        
                        $ spi-filtervol vol1.spi vol2.spi -p params.spi -o vol_0001.spi -r 15
                        
                        http://guam/vispider/vispider/manual.html#module-vispider.batch.filter_volume
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
        max_filename_len = 78,
    )
if __name__ == "__main__": main()






